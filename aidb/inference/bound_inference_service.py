import asyncio
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import sqlalchemy.ext.asyncio
from sqlalchemy.schema import ForeignKeyConstraint

from aidb.config.config_types import Column, InferenceBinding, Table
from aidb.inference.inference_service import InferenceService
from aidb.utils.asyncio import asyncio_run
from aidb.utils.constants import cache_table_name_from_inputs
from aidb.utils.logger import logger


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


  async def _insert_in_cache_table(self, row, conn):
    input_dic = {}
    for col in self.binding.input_columns:
      input_dic[self.convert_normalized_col_name_to_cache_col_name(col)] = getattr(row, col)
    insert = self.get_insert()(self._cache_table).values(**input_dic)
    await conn.execute(insert)


  async def infer(self, inputs: pd.DataFrame):
    # FIXME: figure out where to put the column renaming
    for idx, col in enumerate(self.binding.input_columns):
      inputs.rename(columns={inputs.columns[idx]: col}, inplace=True)

    # Note: the input columns are assumed to be in order
    query_futures = []
    async with self._engine.begin() as conn:
      for ind, row in inputs.iterrows():
        cache_query = self._cache_query_stub.where(
          sqlalchemy.sql.and_(
            *[col == row[self.convert_cache_column_name_to_normalized_column_name(col.name)] for idx, col in enumerate(self._cache_columns)]
          )
        )
        query_futures.append(conn.execute(cache_query))
      query_futures = await asyncio.gather(*query_futures)
      is_in_cache = [len(result.fetchall()) > 0 for result in query_futures]

    results = []
    # TODO
    # - Batch the service inference
    # - Batch the inserts for new data
    # - Batch the selection for cached results... How to do this?
    async with self._engine.begin() as conn:
      for idx, (_, inp_row) in enumerate(inputs.iterrows()):
        if is_in_cache[idx]:
          query = self._result_query_stub.where(
            sqlalchemy.sql.and_(
              *[getattr(self._cache_table.c, self.convert_normalized_col_name_to_cache_col_name(col)) == getattr(inp_row, col) for col in
                self.binding.input_columns]
            )
          )
          df = await conn.run_sync(lambda conn: pd.read_sql(query, conn))
          results.append(df)
        else:
          inference_results = self.service.infer_one(inp_row)

          if len(inference_results) > 0:
            # FIXME: figure out where to put the column renaming
            for idx, col in enumerate(self.binding.output_columns):
              inference_results.rename(columns={inference_results.columns[idx]: col}, inplace=True)

            try:
              # returned results may have few redundant columns because of copying input
              inference_results = inference_results[list(self.binding.output_columns)]
            except:
              raise Exception("Column binding column not found in the inference results")

            tables = self.get_tables(self.binding.output_columns)
            for table in tables:
              columns = [col for col in self.binding.output_columns if col.startswith(table + '.')]
              tmp_df = inference_results[columns]
              tmp_df = tmp_df.astype('object')

              if self._dialect == 'mysql' or self._dialect == 'postgresql':
                tmp_values = tmp_df.to_dict(orient='list')
                values = {}
                for k, v in tmp_values.items():
                  k = k.split('.')[1]
                  values[k] = v
                insert = self.get_insert()(self._tables[table]._table).values(**values)

                # FIXME: does this need to be used anywhere else?
                # FIXME: needs to be tested for sqlite and postgresql
                if len(self._tables[table].primary_key) > 0:
                  if self._dialect == 'mysql':
                    insert = insert.on_duplicate_key_update(
                      values
                    )
                  elif self._dialect == 'postgresql':
                    insert = insert.on_conflict_do_update(
                      index_elements=self._tables[table].primary_key,
                      set_=values,
                    )
                await conn.execute(insert)
              elif self._dialect == 'sqlite':
                for idx, row in tmp_df.iterrows():
                  sqlalchemy_row = {}
                  for col in tmp_df.columns:
                    col_name = col.split('.')[1]
                    sqlalchemy_row[col_name] = row[col]
                  # on_conflict_do_update takes only 1 row at a time
                  insert = self.get_insert()(self._tables[table]._table).values(sqlalchemy_row).on_conflict_do_update(
                    index_elements=self._tables[table].primary_key,
                    set_=sqlalchemy_row,
                  )
                  await conn.execute(insert)

          await self._insert_in_cache_table(inp_row, conn)
          results.append(inference_results)

    return results


  def __hash__(self):
    return super().__hash__()