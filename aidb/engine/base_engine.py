from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap

from aidb.config.config import Config
from aidb.config.config_types import InferenceBinding
from aidb.inference.bound_inference_service import (
    BoundInferenceService, CachedBoundInferenceService)
from aidb.inference.inference_service import InferenceService
from aidb.query.query import FilteringClause, Query
from aidb.query.utils import predicate_to_str
from aidb.utils.asyncio import asyncio_run
from aidb.utils.logger import logger


class BaseEngine():
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      debug: bool = False,
  ):
    self._connection_uri = connection_uri
    self._debug = debug

    self._dialect = self._infer_dialect(connection_uri)
    self._sql_engine = self._create_sql_engine()

    if infer_config:
      self._config: Config = asyncio_run(self._infer_config())


  def __del__(self):
    asyncio_run(self._sql_engine.dispose())


  # ---------------------
  # Setup
  # ---------------------
  def _infer_dialect(self, connection_uri: str):
    # Conection URIs have the following format:
    # dialect+driver://username:password@host:port/database
    # See https://docs.sqlalchemy.org/en/20/core/engines.html
    dialect = connection_uri.split(':')[0]
    if '+' in dialect:
      dialect = dialect.split('+')[0]

    supported_dialects = [
      'mysql',
      'postgresql',
      'sqlite',
    ]

    if dialect not in supported_dialects:
      logger.warning(
        f'Unsupported dialect: {dialect}. Defaulting to mysql')
      dialect = 'mysql'

    return dialect


  def _create_sql_engine(self):
    logger.info(f'Creating SQL engine for {self._dialect}')
    if self._dialect == 'mysql':
      kwargs = {
        'echo': self._debug,
        'max_overflow': -1,
      }
    else:
      kwargs = {}

    engine = sqlalchemy.ext.asyncio.create_async_engine(
      self._connection_uri,
      **kwargs,
    )

    return engine


  async def _infer_config(self) -> Config:
    '''
    Infer the database configuration from the sql engine.
    Extracts:
    - Tables, columns (+ types), and foriegn keys.
    - Cache tables
    - Blob tables
    - Generated columns
    '''

    # We use an async engine, so we need a function that takes in a synchrnous connection
    def config_from_conn(conn):
      config = Config(
        {},
        [],
        self._connection_uri,
        None,
        None,
        None,
        None,
        None,
      )
      config.load_from_sqlalchemy(conn)
      return config

    async with self._sql_engine.begin() as conn:
      config: Config = await conn.run_sync(config_from_conn)

    if self._debug:
      import prettyprinter as pp
      pp.install_extras(
        exclude=['django', 'ipython', 'ipython_repr_pretty'])
      pp.pprint(config)
      print(config.blob_tables)

    return config


  def register_inference_service(self, service: InferenceService):
    self._config.add_inference_service(service.name, service)


  def bind_inference_service(self, service_name: str, binding: InferenceBinding):
    bound_service = CachedBoundInferenceService(
      self._config.inference_services[service_name],
      binding,
      self._sql_engine,
      self._config.columns,
      self._config.tables,
      self._dialect,
    )
    self._config.bind_inference_service(bound_service)


  # ---------------------
  # Properties
  # ---------------------
  @property
  def dialect(self):
    return self._dialect


  # ---------------------
  # Inference
  # ---------------------
  def prepare_multitable_inputs(self, raw_inputs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    '''
    Prepare the inputs for inference.
    '''
    assert len(raw_inputs) >= 1
    final_df = raw_inputs[0][1]
    for idx, (table_name, df) in enumerate(raw_inputs[1:]):
      last_table_name = raw_inputs[idx][0]
      table_relations = self._config.relations_by_table[table_name]
      join_keys = [
        fk for fk in table_relations if fk.startswith(last_table_name)]
      final_df = final_df.merge(df, on=join_keys, how='inner')

    return final_df


  def process_inference_outputs(self, binding: InferenceBinding, joined_outputs: pd.DataFrame) -> pd.DataFrame:
    '''
    Process the outputs of inference by renaming the columns and selecting the
    output columns.
    '''
    df_cols = list(joined_outputs.columns)
    for idx, col in enumerate(binding.output_columns):
      joined_outputs.rename(columns={df_cols[idx]: col}, inplace=True)
    res = joined_outputs[list(binding.output_columns)]
    return res


  def inference(self, inputs: pd.DataFrame, bound_service: BoundInferenceService) -> List[pd.DataFrame]:
    return bound_service.batch(inputs)


  def execute(self, query: str):
    raise NotImplementedError()


  def _find_join_path(
      self,
      common_columns: Dict[Tuple[str, str], List[str]],
      table_relations: Dict[str, List[str]],
      table_names: List[str]
  ) -> str:
    """
    Find the path to join tables based on common columns and create the JOIN part of an SQL query.
    :param common_columns: Dict containing common column names between table pairs.
    :param table_relations: Dict containing related tables.
    :param table_names: List of table names to be joined.
    """
    join_strs = []
    stack = [table_names[0]]
    visited = {table_names[0]}
    while stack:
      current_table = stack.pop()
      for neighbor_table in table_relations[current_table]:
        if neighbor_table in visited:
          continue
        visited_col = []
        join_condition = []
        for visited_table in visited:
          for col in common_columns[(neighbor_table, visited_table)]:
            if col not in visited_col:
              join_condition.append(
                f'{visited_table}.{col} = {neighbor_table}.{col}')
              visited_col.append(col)
        join_strs.append(
          f'INNER JOIN {neighbor_table} ON {" AND ".join(join_condition)}')
        visited.add(neighbor_table)
        stack.append(neighbor_table)
    return f"FROM {table_names[0]}\n" + '\n'.join(join_strs)


  def _get_tables(self, columns: List[str]) -> List[str]:
    tables = set()
    for col in columns:
      table_name = col.split('.')[0]
      tables.add(table_name)
    return list(tables)


  def _get_inner_join_query(self, table_names: List[str]):
    """
    Generate an SQL query to perform INNER JOIN operations on the provided tables.
    :param selected_cols: List of column names to be selected in the SQL query.
    :param table_names: List of table names to be joined.
    """
    table_number = len(table_names)
    table_relations = defaultdict(list)
    common_columns = {}
    for i in range(table_number - 1):
      table1 = self._config.tables[table_names[i]]
      table1_cols = [col.name for col in table1.columns]
      for j in range(i + 1, table_number):
        table2 = self._config.tables[table_names[j]]
        table2_cols = [col.name for col in table2.columns]
        common = list(set(table1_cols).intersection(table2_cols))

        common_columns[(table_names[i], table_names[j])] = common
        common_columns[(table_names[j], table_names[i])] = common
        if common:
          table_relations[table_names[i]].append(table_names[j])
          table_relations[table_names[j]].append(table_names[i])

    join_path_str = self._find_join_path(
      common_columns, table_relations, table_names)
    return join_path_str


  def _get_where_str(self, filtering_predicates: List[List[FilteringClause]]):
    and_connected = []
    for fp in filtering_predicates:
      and_connected.append(" OR ".join(
        [predicate_to_str(p) for p in fp]))
    return " AND ".join(and_connected)


  def get_input_query_for_inference_service(
      self,
      bound_service: BoundInferenceService,
      user_query,
      rep_table_name: Optional[str] = None,
      filtered_index_list: Optional[List[int]] = None
  ):
    """
    this function returns the input query to fetch the input records for an inference service
    input query will also contain the predicates that can be currently satisfied using the inference services
    that are already executed
    """

    column_to_root_column = self._config.columns_to_root_column
    inp_cols = bound_service.binding.input_columns
    root_inp_cols = [column_to_root_column.get(
      col, col) for col in inp_cols]

    if rep_table_name:
      root_inp_cols.append(f'{rep_table_name}.blob_id')

    inp_cols_str = ', '.join(root_inp_cols)
    inp_tables = self._get_tables(root_inp_cols)
    join_str = self._get_inner_join_query(inp_tables)

    service_name = bound_service.service.name
    filtering_predicates = user_query.inference_engines_related_filtering_predicates[service_name]
    filtering_predicates_satisfied = user_query.get_satisfied_filtering_predicates(filtering_predicates, inp_tables)

    where_str = self._get_where_str(filtering_predicates_satisfied)
    for k, v in column_to_root_column.items():
      where_str = where_str.replace(k, v)

    # FIXME: for different database, the IN grammar maybe different
    if filtered_index_list:
      filtered_index_list = tuple(filtered_index_list)
      filter_condition = f"{rep_table_name}.blob_id IN {format(filtered_index_list)}"
      where_str = f"{where_str} AND {filter_condition}" if where_str else filter_condition

    where_clause = f"WHERE {where_str}" if where_str else ""
    inp_query_str = f'''
                SELECT {inp_cols_str}
                {join_str}
                {where_clause};
            '''
    return inp_query_str

