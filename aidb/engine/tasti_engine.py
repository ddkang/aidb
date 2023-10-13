import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
from typing import Dict, List, Optional

from aidb.config.config_types import Table
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.query.query import Query
from aidb.query.utils import predicate_to_str
from aidb.utils.constants import table_name_for_rep_and_topk
from aidb.vector_database.tasti import Tasti
from aidb.vector_database.vector_database_config import TastiConfig


class TastiEngine(FullScanEngine):
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      debug: bool = False,
      blob_mapping_table_name: Optional[str] = None,
      tasti_config: Optional[TastiConfig] = None,
  ):
    super().__init__(connection_uri, infer_config, debug)

    self.rep_table_name = None
    # TODO: modify to same rep table with different topk table
    self.topk_table_name = None
    # table for mapping blob keys to blob ids
    self.blob_mapping_table_name = blob_mapping_table_name
    self.tasti_config = tasti_config


  async def execute_tasti(self, query: Query, **kwargs):
    bound_service_list = self._get_required_bound_services_order(query)

    blob_tables = set()
    for bound_service in bound_service_list:
      for input_col in bound_service.binding.input_columns:
        input_table = input_col.split('.')[0]
        if input_table in self._config.blob_tables:
          blob_tables.add(input_table)
    blob_tables = list(blob_tables)

    self.rep_table_name, self.topk_table_name = table_name_for_rep_and_topk(blob_tables)
    reps_df = await self._get_cluster_rep_blob_ids()
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service(bound_service, query,
                                                                 self.rep_table_name, list(reps_df.index))
      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
      inp_df.set_index('blob_id', inplace=True, drop=True)
      reps_df = reps_df.loc[inp_df.index]
      await bound_service.infer(inp_df)

    score_query_str, score_connected = self._get_score_query_str(query, self.rep_table_name)
    async with self._sql_engine.begin() as conn:
      score_df = await conn.run_sync(lambda conn: pd.read_sql(text(score_query_str), conn))
    score_df.set_index('blob_id', inplace=True, drop=True)
    score_for_all_df = await self.get_score_for_all_blob_ids(score_df)
    proxy_score_for_all_blobs = self.score_fn(score_for_all_df, score_connected)

     # FIXME: decide what to return
    return reps_df, proxy_score_for_all_blobs


  async def _get_cluster_rep_blob_ids(self) -> pd.DataFrame:
    if self.rep_table_name not in self._config.tables:
      await self.initialize_tasti()

    reps_query_str = f'''
                SELECT {self.rep_table_name}.blob_id
                FROM {self.rep_table_name};
                '''

    async with self._sql_engine.begin() as conn:
      reps_df = await conn.run_sync(lambda conn: pd.read_sql(text(reps_query_str), conn))
    reps_df.set_index('blob_id', inplace=True, drop=True)

    return reps_df


  def _get_filter_predicate_score_map(self, query: Query) -> (Dict[str, str], List[List[str]]):
    '''
    Convert WHERE filtering predicate into 0/1 score, record the relation between different filtering predicate
    '''
    filering_predicate_score_map = dict()
    and_connected = []
    score_count = 0
    for fp in query.filtering_predicates:
      or_connected = []
      for p in fp:
        predicate_str = predicate_to_str(p)
        if predicate_str not in filering_predicate_score_map:
          filering_predicate_score_map[predicate_str] = f'score_{score_count}'
          score_count += 1
        or_connected.append(filering_predicate_score_map[predicate_str])
      and_connected.append(or_connected)

    return filering_predicate_score_map, and_connected


  def _get_score_query_str(self, query: Query, rep_table_name: str) -> (str, List[List[str]]):
    filering_predicate_score_map, score_connected = self._get_filter_predicate_score_map(query)
    score_list = []
    # FIXME: for different database, the IN grammar maybe different
    for fp in filering_predicate_score_map:
      score_list.append(f'IIF({fp}, 1, 0) AS {filering_predicate_score_map[fp]}')
    score_list.append(f'{rep_table_name}.blob_id')
    select_str = ', '.join(score_list)
    cols = list(filering_predicate_score_map.keys())
    cols.append(f'{rep_table_name}.blob_id')
    tables = self._get_tables(cols)
    join_str = self._get_inner_join_query(tables)
    score_query_str = f'''
                      SELECT {select_str}
                      {join_str}
                      '''
    return score_query_str, score_connected


  async def get_score_for_all_blob_ids(self, score_df: pd.DataFrame, return_binary_score = False) -> pd.DataFrame:

    topk_query_str = f'SELECT * FROM {self.topk_table_name}'
    async with self._sql_engine.begin() as conn:
      topk_df = await conn.run_sync(lambda conn: pd.read_sql(text(topk_query_str), conn))
    topk_df.set_index('blob_id', inplace=True, drop=True)

    topk = len(topk_df.columns) // 2
    topk_indices = np.arange(topk)

    score_df_cols = score_df.columns
    all_dists = np.zeros((len(topk_df), topk))
    all_scores = np.zeros((len(topk_df), len(score_df_cols), topk))

    for idx, (index, row) in enumerate(topk_df.iterrows()):
      reps = [int(row[f'topk_reps_{i}']) for i in topk_indices]
      dists = [row[f'topk_dists_{i}'] for i in topk_indices]

      if index in score_df.index:
        reps = [index] * topk
        dists = [1] * topk

      all_dists[idx, :] = dists
      for col_idx, col_name in enumerate(score_df_cols):
        all_scores[idx, col_idx, :] = score_df[col_name].loc[reps].values

    # to avoid division by zero error
    all_dists += 1e-8

    if return_binary_score:
      weights = 1.0 / all_dists
      weights = weights[:, np.newaxis, :]
      votes_1 = np.sum(all_scores * weights, axis=2)
      votes_0 = np.sum((1 - all_scores) * weights, axis=2)
      # majority vote
      y_pred = (votes_1 > votes_0).astype(int)
    else:
      weights = np.sum(all_dists, axis=1).reshape(-1, 1) - all_dists
      weights = weights / weights.sum(axis=1).reshape(-1, 1)
      weights = weights[:, np.newaxis, :]
      y_pred = np.sum(all_scores * weights, axis=2)

    return pd.DataFrame(y_pred, columns=score_df_cols, index=topk_df.index)


  def score_fn(self, score_for_all_df: pd.DataFrame, score_connected: List[List[str]]) -> pd.Series:
    '''
    convert query result to score, return a Dataframe contains score column, the index is blob index
    if A and B, then the score is min(score A, score B)
    if A or B, then the score is max(score A, score B)
    '''
    proxy_score_all_blobs = np.zeros(len(score_for_all_df))
    for idx, (index, row) in enumerate(score_for_all_df.iterrows()):
      min_score = 1
      for or_connected in score_connected:
        max_score = 0
        for score_name in or_connected:
          max_score = max(max_score, row[score_name])
        min_score = min(min_score, max_score)
      proxy_score_all_blobs[idx] = min_score
    return pd.Series(proxy_score_all_blobs, index=score_for_all_df.index)


  def _create_tasti_table(self, topk:int, conn: sqlalchemy.engine.base.Connection):
    '''
    create new rep_table and topk_table to store the results from vector database.
    rep_table is used to store cluster representatives ids and blob keys,
    topk_table is used to store topk rep_ids and dists for all blobs.
    '''
    metadata = sqlalchemy.MetaData()
    metadata.reflect(conn)
    blob_mapping_table = sqlalchemy.schema.Table(self.blob_mapping_table_name, metadata,
                                                 autoload=True, autoload_with=conn)

    # FIXME: there is a sqlalchemy SAWarning, This warning may become an exception in a future release
    rep_table = sqlalchemy.schema.Table(self.rep_table_name, metadata,
                                        *[column._copy() for column in blob_mapping_table.columns],
                                        *[constraint._copy() for constraint in blob_mapping_table.constraints]
                                        )

    columns = []
    columns.append(sqlalchemy.Column(f'blob_id', sqlalchemy.Integer, primary_key=True, autoincrement=False))
    for i in range(topk):
      columns.append(sqlalchemy.Column(f'topk_reps_{str(i)}', sqlalchemy.Integer))
      columns.append(sqlalchemy.Column(f'topk_dists_{str(i)}', sqlalchemy.Float))
    topk_table = sqlalchemy.schema.Table(self.topk_table_name, metadata, *columns)
    metadata.create_all(conn)

    self._config.tables[self.rep_table_name] = Table(rep_table)
    self._config.tables[self.topk_table_name] = Table(topk_table)


  def _format_topk_for_all(self, topk_for_all: pd.DataFrame) -> pd.DataFrame:
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


  async def initialize_tasti(self):
    """
    This function initializes the Tasti index and manages the insertion of data into
    representative and topk tables.
    It performs the following main steps:
    1. Initializes a Tasti index and retrieves representative blob ids and topk reps and dists.
    2. Creates Tasti tables and retrieves culster representative blob key columns.
    3. Inserts data into the representative table and topk table
    """
    if self.tasti_config is None:
      raise Exception('TASTI hasn\'t been initialized, please provide TASTI config')
    tasti_index = Tasti(self.tasti_config)
    rep_ids = tasti_index.get_representative_blob_ids()
    topk_for_all = tasti_index.get_topk_representatives_for_all()
    new_topk_for_all = self._format_topk_for_all(topk_for_all)
    topk = max(topk_for_all['topk_reps'].str.len())

    rep_blob_query_str = f'''
                    SELECT *
                    FROM {self.blob_mapping_table_name}
                    WHERE {self.blob_mapping_table_name}.blob_id IN {format(tuple(rep_ids.index))}
                    '''

    async with self._sql_engine.begin() as conn:
      await conn.run_sync(lambda conn: self._create_tasti_table(topk, conn))
      rep_blob_df = await conn.run_sync(lambda conn: pd.read_sql(text(rep_blob_query_str), conn))
      # FIXME: same as db_setup.py line 147, in case of mysql,
      #  this function doesn't wait, hence throwing integrity error
      await conn.run_sync(lambda conn: rep_blob_df.to_sql(self.rep_table_name, conn, if_exists='append', index=False))
      await conn.run_sync(lambda conn: new_topk_for_all.to_sql(self.topk_table_name, conn,
                                                               if_exists='append', index=False))

  # TODO: update topk table and representative table, query for new embeddings.
