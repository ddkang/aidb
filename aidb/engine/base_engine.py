import collections
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap
from sqlalchemy.sql import text
from sqlalchemy.schema import ForeignKeyConstraint

from aidb.config.config import Config
from aidb.config.config_types import Graph, InferenceBinding, TastiConfig, Table
from aidb.inference.bound_inference_service import (
    BoundInferenceService, CachedBoundInferenceService)
from aidb.inference.inference_service import InferenceService
from aidb.utils.asyncio import asyncio_run
from aidb.utils.logger import logger
from aidb.vector_database.tasti import Tasti
from aidb.utils.constants import table_name_for_rep_and_topk

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
      logger.warning(f'Unsupported dialect: {dialect}. Defaulting to mysql')
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
      pp.install_extras(exclude=['django', 'ipython', 'ipython_repr_pretty'])
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
      join_keys = [fk for fk in table_relations if fk.startswith(last_table_name)]
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


  def get_left_join_str(self, rep_table_name: str, tables: List[str]) -> str:
    """
    Constructs a LEFT JOIN SQL string based on the given representative table name and a list of table names.
    :param rep_table_name: Name of the representative table.
    :param tables: List of table names to join.
    """
    join_strs = []
    rep_cols = [rep_col.split('.')[0] for rep_col in self._config.tables[rep_table_name].columns]
    for table_name in tables:
      table_cols = [table_col.split('.')[0] for table_col in self._config.tables[table_name].columns]
      join_conditions = [f"{rep_table_name}.{col} = {table_name}.{col}" for col in table_cols if col in rep_cols]
      if join_conditions:
        join_str = f"LEFT JOIN {table_name} ON {' AND '.join(join_conditions)}"
        join_strs.append(join_str)
      else:
        raise Exception(f'Can\'t join table {rep_table_name} and {table_name}')
    return f"FROM {rep_table_name}\n" + '\n'.join(join_strs)


  async def get_data(self, rep_table_name: str, columns: Tuple[str]) -> pd.DataFrame:
    """
    Retrieves input or output data for cluster representatives.
    :param rep_table_name: Name of the representative table.
    :param columns: List of columns to be selected.
    """
    cols_str = ', '.join(columns)
    tables = self.get_tables(columns)
    join_str = self.get_join_str(rep_table_name, tables)
    query_str = f'''
            SELECT {cols_str}
            {join_str};
          '''

    async with self._sql_engine.begin() as conn:
      inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str), conn))
    return inp_df


  def get_score_for_all_blob_ids(
      self,
      score_fn,
      bound_service: BoundInferenceService,
      tasti_config: Optional[TastiConfig] = None
  ) -> pd.Series:
    '''
    Get representatives id, query the normal DB to get the output of target labeler.
    Based on user provided function, calculate the score for representatives id
    '''
    blob_tables = {}
    for input_col in bound_service.binding.input_columns:
      input_table = input_col.split('.')[0]
      if input_table in self._config.blob_tables:
        blob_tables.add(input_table)

    rep_table_name, topk_table_name = table_name_for_rep_and_topk(list(blob_tables))
    if rep_table_name not in self._config.tables:
      if tasti_config is None:
        raise Exception('TASTI hasn\'t been initialized, please provide TASTI config')
      self.initialize_tasti(tasti_config, bound_service)
      inp_df = asyncio_run(self.get_data(rep_table_name, bound_service.binding.input_columns))
      self.inference(inp_df, bound_service)

    out_df = self.get_data(rep_table_name, bound_service.binding.output_columns)

    #FIXME: use user defined function to get score for each blob id, index should be blob id and one column for score.
    true_score = score_fn(out_df)
    topk_df = asyncio_run(self.get_topk_table(topk_table_name))
    return self._calculate_score_for_all_blob_ids(true_score, topk_df)


  def _calculate_score_for_all_blob_ids(
      self,
      true_score: pd.DataFrame,
      topk_df: pd.DataFrame,
      return_binary_score=False
  ) -> pd.Series:

    topk = len(topk_df.columns) // 2
    all_dists, all_scores = [], []
    for index, row in topk_df.iterrows():
      reps = [int(row[f'topk_reps_{i}']) for i in range(topk)]
      dists = [row[f'topk_dists_{i}'] for i in range(topk)]
      if row.name in true_score.index:
        reps = [row.name] * topk
        dists = [0] * topk
      all_dists.append(dists)
      all_scores.append(true_score.loc[reps].values.flatten())

    # to avoid division by zero error
    all_dists = np.array(all_dists) + 1e-8
    all_scores = np.array(all_scores)
    all_dists += 1e-8
    if return_binary_score:
      weights = 1.0 / all_dists
      votes_1 = np.sum(all_scores * weights, axis=1)
      votes_0 = np.sum((1 - all_scores) * weights, axis=1)
      # majority vote
      y_pred = (votes_1 > votes_0).astype(int)
      return pd.Series(y_pred, index=topk_df.index)
    else:
      weights = np.sum(all_dists, axis=1).reshape(-1, 1) - all_dists
      weights = weights / weights.sum(axis=1).reshape(-1, 1)
      y_pred = np.sum(all_scores * weights, axis=1)
      return pd.Series(y_pred, index=topk_df.index)


  async def get_topk_table(self, topk_table_name: str):
    '''
    retrieve the topk reps and dists for all blobs from table
    '''
    async with self._sql_engine.begin() as conn:
      query = f''' SELECT * FROM {topk_table_name}'''
      topk_df = await conn.run_sync(lambda conn: pd.read_sql(text(query), conn))
      topk_df.set_index('blob_id', inplace=True, drop=True)
      return topk_df


  def find_join_path(
      self,
      common_columns: Dict[Tuple[str,str], List[str]],
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
    visited = set(table_names[0])
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
              join_condition.append(f'{visited_table}.{col} = {neighbor_table}.{col}')
              visited_col.append(col)
        join_strs.append(f'INNER JOIN {neighbor_table} ON {" AND ".join(join_condition)}')
        visited.add(neighbor_table)
        stack.append(neighbor_table)
    return f"FROM {table_names[0]}\n" + '\n'.join(join_strs)


  def get_inner_join_query(self, selected_cols: List[str], table_names: List[str]):
    """
    Generate an SQL query to perform INNER JOIN operations on the provided tables.
    :param selected_cols: List of column names to be selected in the SQL query.
    :param table_names: List of table names to be joined.
    """
    table_number = len(table_names)
    table_relations = collections.defaultdict(list)
    common_columns = {}
    for i in range(table_number - 1):
      table1 = self._config.tables[table_names[i]]
      table1_cols = [col.split('.')[1] for col in table1.columns]
      for j in range(i+1, table_number):
        table2 = self._config.tables[table_names[j]]
        table2_cols = [col.split('.')[1] for col in table2.columns]
        common = list(set(table1_cols).intersection(table2_cols))

        common_columns[(table_names[i], table_names[j])] = common
        common_columns[(table_names[j], table_names[i])] = common
        if common:
          table_relations[table_names[i]].append(table_names[j])
          table_relations[table_names[j]].append(table_names[i])

    join_path_str = self.find_join_path(common_columns, table_relations, table_names)
    cols_str = ', '.join(selected_cols)
    query_str = f'''
                SELECT {cols_str}
                {join_path_str};
                '''
    return query_str


  async def get_blob_key_data(self, selected_cols: List[str], table_names: List[str]):

    query_str = self.get_inner_join_query(selected_cols, table_names)
    async with self._sql_engine.begin() as conn:
      blob_key_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str), conn))
    return blob_key_df



  async def _create_tasti_table(self, blob_tables: List[str], topk: int):
    '''
    create new rep_table and topk_table to store the results from vector database.
    rep_table is used to store cluster representatives ids and blob keys,
    topk_table is used to store topk reps_ids and dists for all blobs.
    '''
    all_blob_key = []
    all_col_name = []
    rep_table_name, topk_table_name = table_name_for_rep_and_topk(blob_tables)

    async with self._sql_engine.begin() as conn:
      new_metadata = sqlalchemy.MetaData()
      columns = []
      fk_constraints = {}
      columns.append(sqlalchemy.Column(f'blob_id', sqlalchemy.Integer, primary_key=True, autoincrement=False))
      for blob_table in blob_tables:
        blob_keys = self._config.blob_keys[blob_table]
        fk_constraints[blob_table] = {'cols': [], 'cols_refs': []}
        for blob_key in blob_keys:
          col_name = blob_key.split('.')[1]
          if col_name not in all_col_name:
            all_blob_key.append(blob_key)
            all_col_name.append(col_name)
            columns.append(sqlalchemy.Column(col_name, self._config.columns[blob_key].type))
            fk_constraints[blob_table]['cols'].append(col_name)
            fk_constraints[blob_table]['cols_refs'].append(blob_key)

      multi_table_fk_constraints = []
      for tbl, fk_cons in fk_constraints.items():
        multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))
      rep_table = sqlalchemy.schema.Table(rep_table_name, new_metadata, *columns, *multi_table_fk_constraints)

      columns = []
      columns.append(sqlalchemy.Column(f'blob_id', sqlalchemy.Integer, primary_key=True, autoincrement=False))
      for i in range(topk):
        columns.append(sqlalchemy.Column(f'topk_reps_{str(i)}', sqlalchemy.Integer))
        columns.append(sqlalchemy.Column(f'topk_dists_{str(i)}', sqlalchemy.Float))
      topk_table = sqlalchemy.schema.Table(topk_table_name, new_metadata, *columns)
      await conn.run_sync(new_metadata.create_all)

      self._config.tables[rep_table_name] = Table(rep_table)
      self._config.tables[topk_table] = Table(topk_table)

    return all_blob_key


  async def insert_data(self, data: pd.DataFrame, table_name: str):
    async with self._sql_engine.begin() as conn:
      table = self._config.tables[table_name]
      insert = self.get_insert()
      await conn.execute(insert(table), data.to_dict(orient='records'))


  def initialize_tasti(self, tasti_config: TastiConfig, bound_service):
    """
    This function initializes the Tasti index and manages the insertion of data into
    representative and topk tables.
    It performs the following main steps:
    1. Initializes a Tasti index and retrieves representative blob ids and topk reps and dists.
    2. Creates Tasti tables and retrieves culster representative blob key columns.
    3. Concatenates representative ids and blob key data, then inserts into the representative table.
    4. Formats the topk reps and dists and inserts into the topk table.
    Notice: we assume the data in blob table has same order as its embedding in vector database
    """
    tasti_index = Tasti(tasti_config)
    rep_ids = tasti_index.get_representative_blob_ids()
    topk_for_all = tasti_index.vector_database.get_topk_representatives_for_all()

    blob_tables = {}
    for input_col in bound_service.binding.input_columns:
      input_table = input_col.split('.')[0]
      if input_table in self._config.blob_tables:
        blob_tables.add(input_table)
    blob_tables = list(blob_tables)
    rep_table_name, topk_table_name = table_name_for_rep_and_topk(blob_tables)

    all_blob_key_cols = asyncio_run(self._create_tasti_table(blob_tables))

    blob_key_data = asyncio_run(self.get_blob_key_data(all_blob_key_cols, blob_tables))
    selected_blob_key_data = blob_key_data.ioc[rep_ids['blob_id'].values.tolist()]
    rep_data = pd.concat([rep_ids.reset_index(drop=True), selected_blob_key_data.reset_index(drop=True)], axis=1)
    asyncio_run(self.insert_data(rep_data, rep_table_name))

    new_topk_for_all = self._format_topk_for_all(topk_for_all)
    asyncio_run(self.insert_data(new_topk_for_all, topk_table_name))


  def _format_topk_for_all(self, topk_for_all: pd.DataFrame):
    """
    Formats the top K for all data.
    :param topk_for_all: Top K for all data.
    """
    topk = max(topk_for_all['topk_reps'].str.len())
    new_topk_for_all = {
      f'topk_reps_{i}': topk_for_all['topk_reps'].str[i]
      for i in range(topk)
    }
    new_topk_for_all.update({
      f'topk_dists_{i}': topk_for_all['topk_dists'].str[i]
      for i in range(topk)
    })
    return pd.DataFrame(new_topk_for_all)

  #TODO: update topk table and representative table, query for new embeddings.
