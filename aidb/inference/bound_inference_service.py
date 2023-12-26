from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import sqlalchemy.ext.asyncio
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import text

from aidb.config.config_types import Column, InferenceBinding, Table
from aidb.inference.inference_service import InferenceService
from aidb.utils.asyncio import asyncio_run
from aidb.utils.constants import cache_table_name_from_inputs
from aidb.utils.logger import logger
from aidb.utils.type_conversion import pandas_dtype_to_native_type


@dataclass
class BoundInferenceService():
  service: InferenceService
  binding: InferenceBinding

  def infer(self, inputs):
    raise NotImplementedError()

  def __hash__(self):
    # The binding must be unique
    return hash(self.binding)


@dataclass
class CachedBoundInferenceService(BoundInferenceService):
  _engine: sqlalchemy.ext.asyncio.AsyncEngine
  _columns: Dict[str, Column]
  _tables: Dict[str, Table]
  _dialect: str
  _verbose: bool=False


  def optional_tqdm(self, iterable, **kwargs):
    if self._verbose:
      from tqdm import tqdm
      return tqdm(iterable, **kwargs)
    return iterable


  def convert_normalized_col_name_to_cache_col_name(self, column_name: str):
    return column_name.replace('.', '__')


  def convert_cache_column_name_to_normalized_column_name(self, column_name: str):
    return column_name.replace('__', '.')


  def __post_init__(self):
    self._cache_table_name = cache_table_name_from_inputs(self.service.name, self.binding.input_columns)
    logger.debug('Cache table name', self._cache_table_name)

    # Get table takes a synchronous connection
    def get_table(conn: sqlalchemy.engine.base.Connection):
      inspector = sqlalchemy.inspect(conn)
      metadata = sqlalchemy.MetaData()

      columns = []
      fk_constraints = {}
      for column_name in self.binding.input_columns:
        column = self._columns[column_name]
        new_table_col_name = self.convert_normalized_col_name_to_cache_col_name(column_name)
        logger.debug('New table col name', new_table_col_name)
        logger.debug('Ref column', str(column))
        fk_ref_table_name = str(column).split('.')[0]
        if fk_ref_table_name not in fk_constraints:
          fk_constraints[fk_ref_table_name] = {'cols': [], 'cols_refs': []}
        # both tables will have same column name
        fk_constraints[fk_ref_table_name]['cols'].append(new_table_col_name)
        fk_constraints[fk_ref_table_name]['cols_refs'].append(column)
        columns.append(sqlalchemy.schema.Column(new_table_col_name, column.type))

      multi_table_fk_constraints = []
      for _, fk_cons in fk_constraints.items():
        multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))

      table = sqlalchemy.schema.Table(self._cache_table_name, metadata, *columns, *multi_table_fk_constraints)
      # Create the table if it doesn't exist
      if not self._cache_table_name in inspector.get_table_names():
        metadata.create_all(conn)

      return table, columns

    async def tmp():
      async with self._engine.begin() as conn:
        return await conn.run_sync(get_table)

    self._cache_table, self._cache_columns = asyncio_run(tmp())

    # Query the table to see if it works
    async def query_table():
      async with self._engine.begin() as conn:
        tmp = await conn.execute(self._cache_table.select().limit(1))
        tmp.fetchall()

    try:
      asyncio_run(query_table())
    except:
      raise ValueError(f'Could not query table {self._cache_table_name}')

    self._cache_query_stub = sqlalchemy.sql.select(self._cache_columns)

    output_tables = set()
    output_cols_with_label = []
    for col_name in self.binding.output_columns:
      col = self._columns[col_name]
      output_tables.add(str(col.table))
      output_cols_with_label.append(col.label(col_name))

    joined = self._cache_table
    for table_name in output_tables:
      condition = []
      for col in self._tables[table_name].columns:
        for cache_col in self._cache_columns:
          normal_name = cache_col.name.split('__')[1]
          if col.name == normal_name:
            condition.append(getattr(self._cache_table.c, cache_col.name)
                             == getattr(self._tables[table_name]._table.c, col.name))
      joined = sqlalchemy.join(joined, self._tables[table_name]._table, *condition)
    self._result_query_stub = sqlalchemy.sql.select(output_cols_with_label).select_from(joined)


  def get_insert(self):
    dialect = self._dialect
    if dialect == 'sqlite':
      return sqlalchemy.dialects.sqlite.insert
    elif dialect == 'mysql':
      return sqlalchemy.dialects.mysql.insert
    elif dialect == 'postgresql':
      return sqlalchemy.dialects.postgresql.insert
    else:
      raise NotImplementedError(f'Unknown dialect {dialect}')


  def get_tables(self, columns: List[str]) -> List[str]:
    tables = set()
    for col in columns:
      table_name = col.split('.')[0]
      tables.add(table_name)
    return list(tables)


  async def _insert_in_cache_table(self, inp_rows: List[pd.Series], conn):
    if len(inp_rows) > 0:
      inp_rows_df = pd.DataFrame(inp_rows).reset_index(drop=True)
      inp_rows_df.columns = [self.convert_normalized_col_name_to_cache_col_name(col) for col in inp_rows_df.columns]
      # convert the pandas datatype to python native type
      inp_rows_df = inp_rows_df.astype('object')
      # this doesn't support upsert queries
      await conn.run_sync(lambda conn: inp_rows_df.to_sql(self._cache_table.name, conn, if_exists='append', index=False))


  async def _insert_output_results_in_tables(self, output_results_data: List[pd.DataFrame], conn):
    if len(output_results_data) > 0:
      inference_results = pd.concat(output_results_data, ignore_index=True)
      tables = self.get_tables(self.binding.output_columns)
      for table in tables:
        columns = [col for col in self.binding.output_columns if col.startswith(table + '.')]
        tmp_df = inference_results[columns]
        # convert the pandas datatype to python native type
        tmp_df = tmp_df.astype('object')
        tmp_df.columns = [col.split('.')[1] for col in tmp_df.columns]
        # this doesn't support upsert queries
        await conn.run_sync(lambda conn: tmp_df.to_sql(table, conn, if_exists='append', index=False))


  async def _check_inputs_in_cache_table(self, inputs: pd.DataFrame, conn):
    """
    checks the presence of inputs in the cache table
    """
    cache_entries = await conn.run_sync(lambda conn: pd.read_sql_query(text(str(self._cache_query_stub.compile())), conn))
    cache_entries = cache_entries.set_index([col.name for col in self._cache_columns])
    normalized_cache_cols = [self.convert_cache_column_name_to_normalized_column_name(col.name) for col in
                             self._cache_columns]
    is_in_cache = []
    if len(normalized_cache_cols) == 1:
      for ind, row in inputs.iterrows():
        is_in_cache.append(row[normalized_cache_cols[0]] in cache_entries.index)
    else:
      for ind, row in inputs.iterrows():
        is_in_cache.append(tuple([row[col] for col in normalized_cache_cols]) in cache_entries.index)
    return is_in_cache


  async def infer(self, inputs: pd.DataFrame):
    # FIXME: figure out where to put the column renaming
    for idx, col in enumerate(self.binding.input_columns):
      inputs.rename(columns={inputs.columns[idx]: col}, inplace=True)

    # Note: the input columns are assumed to be in order
    async with self._engine.begin() as conn:
      is_in_cache = await self._check_inputs_in_cache_table(inputs, conn)
      results = []
      records_to_insert_in_table = []
      inputs_to_insert_in_cache_table = []
      # TODO
      # - Batch the service inference
      # - Batch the selection for cached results... How to do this?
      for idx, (_, inp_row) in self.optional_tqdm(enumerate(inputs.iterrows()), total=len(inputs)):
        if is_in_cache[idx]:
          query = self._result_query_stub.where(
            sqlalchemy.sql.and_(
              *[getattr(self._cache_table.c, self.convert_normalized_col_name_to_cache_col_name(col)) == pandas_dtype_to_native_type(getattr(inp_row, col)) for col in
                self.binding.input_columns]
            )
          )
          df = await conn.run_sync(lambda conn: pd.read_sql_query(query, conn))
          results.append(df)
        else:
          inference_results = self.service.infer_one(inp_row)
          for idx, col in enumerate(self.binding.output_columns):
            if idx < len(inference_results.columns):
              inference_results.rename(columns={inference_results.columns[idx]: col}, inplace=True)
            else:  # assume later columns are copied from input
              # assume columns with same name in input / output tables are the same
              _, col_n = col.split('.')
              for c in self.binding.input_columns:
                if c.split('.')[1] == col_n:
                  inference_results[col] = inp_row[c]
                  break
          records_to_insert_in_table.append(inference_results)
          results.append(inference_results)
          inputs_to_insert_in_cache_table.append(inp_row)
      await self._insert_in_cache_table(inputs_to_insert_in_cache_table, conn)
      await self._insert_output_results_in_tables(records_to_insert_in_table, conn)
    return results


  def __hash__(self):
    return super().__hash__()
