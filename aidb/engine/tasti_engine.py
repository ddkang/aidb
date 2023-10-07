from typing import List, Optional, Set

import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text

from aidb.config.config_types import Table
from aidb.vector_database.tasti import Tasti
from aidb.utils.constants import table_name_for_rep_and_topk
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.inference.bound_inference_service import BoundInferenceService
from aidb.query.query import Query
from aidb.vector_database.tasti import TastiConfig


class TastiEngine(FullScanEngine):
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      debug: bool = False,
      tasti_config: Optional[TastiConfig] = None
  ):
    super().__init__(connection_uri, infer_config, debug)

    self.bound_service_list = None
    self.reps_table_name = None
    self.topk_table_name = None
    self.tasti_config = tasti_config


  async def execute_tasti(self, query: Query, **kwargs):
    self.bound_service_list = self._get_required_bound_services_order(query)
    blob_tables = self._get_blob_tables()
    self.reps_table_name, self.topk_table_name = table_name_for_rep_and_topk(blob_tables)

    reps_df = await self._get_cluster_rep_blob_index()
    for bound_service in self.bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service(bound_service, query,
                                                                 self.reps_table_name, list(reps_df.index))
      inp_df = await self.execute_query(inp_query_str, set_index=True, index_column='blob_id')
      reps_df = reps_df.loc[inp_df.index]
      executed_blob_index = await self._get_executed_blob_index(bound_service)
      unexecuted_cluster_reps_blob_index = list(set(reps_df.index) - executed_blob_index)
      await bound_service.infer(inp_df.loc[unexecuted_cluster_reps_blob_index])

    res_df = await self.execute_query(text(query.sql_query_text))
    true_score = self.score_fn(res_df)
    proxy_score_for_all_blob_ids = self.get_proxy_score_for_all_blob_ids(true_score)

    # FIXME: decide what to return
    return res_df, proxy_score_for_all_blob_ids


  def score_fn(self, res_df: pd.DataFrame) -> pd.DataFrame:
    '''
    convert query result to score, return a Dataframe contains score column, the index is blob index
    '''
    raise NotImplementedError


  async def _get_cluster_rep_blob_index(self) -> pd.DataFrame:
    if self.reps_table_name not in self._config.tables:
      await self.initialize_tasti()

    reps_query_str = f'''
                SELECT *
                FROM {self.reps_table_name};
                '''

    reps_df = await self.execute_query(reps_query_str, set_index=True, index_column='blob_id')
    return reps_df


  #TODO: move it to full scan engine
  async def _get_executed_blob_index(self, bound_service: BoundInferenceService) -> Set[int]:
    output_columns = list(bound_service.binding.output_columns)
    output_cols_str = f'{self.reps_table_name}.blob_id'
    output_columns.append(output_cols_str)
    output_tables = self._get_tables(output_columns)
    join_str = self._get_inner_join_query(output_tables)
    where_str = [f'{output_col} IS NOT NULL' for output_col in output_columns]
    where_str = ' AND '.join(where_str)
    query_str = f'''
                SELECT {output_cols_str}
                {join_str}
                WHERE {where_str};
                '''
    executed_blob_df = await self.execute_query(query_str, set_index=True, index_column='blob_id')
    return set(executed_blob_df.index)


  def _get_values_in_topk_table(self):
    '''
    retrieve the topk reps and dists for all blobs from table
    '''
    topk_query = f''' SELECT * FROM {self.topk_table_name}'''
    return topk_query


  async def get_proxy_score_for_all_blob_ids(self, true_score: pd.DataFrame, return_binary_score = True) -> pd.Series:

    topk_query = self._get_values_in_topk_table()
    topk_df = await self.execute_query(topk_query, set_index=True, index_column='blob_id')
    y_true_all_blobs = pd.Series(np.zeros(len(topk_df)), index=topk_df.index)
    y_true_all_blobs.loc[true_score.index] = true_score.values
    topk = len(topk_df.columns) // 2
    topk_indices = np.arange(topk)

    all_dists = np.zeros((len(topk_df), topk))
    all_scores = np.zeros((len(topk_df), topk))

    for idx, (index, row) in enumerate(topk_df.iterrows()):
      reps = [int(row[f'topk_reps_{i}']) for i in topk_indices]
      dists = [row[f'topk_dists_{i}'] for i in topk_indices]

      if index in true_score.index:
        reps = [index] * topk
        dists = [0] * topk

      all_dists[idx, :] = dists
      all_scores[idx, :] = y_true_all_blobs.loc[reps].values.flatten()

    # to avoid division by zero error
    all_dists += 1e-8

    if return_binary_score:
      weights = 1.0 / all_dists
      votes_1 = np.sum(all_scores * weights, axis=1)
      votes_0 = np.sum((1 - all_scores) * weights, axis=1)
      # majority vote
      y_pred = (votes_1 > votes_0).astype(int)
    else:
      weights = np.sum(all_dists, axis=1).reshape(-1, 1) - all_dists
      weights = weights / weights.sum(axis=1).reshape(-1, 1)
      y_pred = np.sum(all_scores * weights, axis=1)

    return pd.Series(y_pred, index=topk_df.index)


  async def _create_tasti_table(self, topk: int):
    '''
    create new rep_table and topk_table to store the results from vector database.
    rep_table is used to store cluster representatives ids and blob keys,
    topk_table is used to store topk reps_ids and dists for all blobs.
    '''
    all_col_name = []
    blob_tables = self._get_blob_tables()
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
            all_col_name.append(col_name)
            columns.append(sqlalchemy.Column(col_name, self._config.columns[blob_key].type))
            fk_constraints[blob_table]['cols'].append(col_name)
            fk_constraints[blob_table]['cols_refs'].append(blob_key)
      multi_table_fk_constraints = []
      rep_table = sqlalchemy.schema.Table(self.reps_table_name, new_metadata, *columns, *multi_table_fk_constraints)

      columns = []
      columns.append(sqlalchemy.Column(f'blob_id', sqlalchemy.Integer, primary_key=True, autoincrement=False))
      for i in range(topk):
        columns.append(sqlalchemy.Column(f'topk_reps_{str(i)}', sqlalchemy.Integer))
        columns.append(sqlalchemy.Column(f'topk_dists_{str(i)}', sqlalchemy.Float))
      topk_table = sqlalchemy.schema.Table(self.topk_table_name, new_metadata, *columns)
      await conn.run_sync(new_metadata.create_all)

      self._config.tables[self.reps_table_name] = Table(rep_table)
      self._config.tables[self.topk_table_name] = Table(topk_table)


  async def insert_data(self, data: pd.DataFrame, table_name: str):
    async with self._sql_engine.begin() as conn:
      # FIXME: same as db_setup.py line 147, in case of mysql,
      #  this function doesn't wait, hence throwing integrity error
      await conn.run_sync(lambda conn: data.to_sql(table_name, conn, if_exists='append', index=False))


  def _get_blob_tables(self) -> List[str]:
    blob_tables = set()
    for bound_service in self.bound_service_list:
      for input_col in bound_service.binding.input_columns:
        input_table = input_col.split('.')[0]
        if input_table in self._config.blob_tables:
          blob_tables.add(input_table)
    blob_tables = list(blob_tables)
    return blob_tables


  def _get_blob_key_query(self) -> str:
    blob_tables = self._get_blob_tables()
    all_blob_keys = []
    for blob_table in blob_tables:
      blob_keys = self._config.blob_keys[blob_table]
      for blob_key in blob_keys:
        all_blob_keys.append(blob_key)

    cols_str = ', '.join(all_blob_keys)
    join_str = self._get_inner_join_query(blob_tables)
    blob_key_query = f'''
                SELECT {cols_str}
                {join_str}'''
    return blob_key_query


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
    3. Concatenates representative ids and blob key data, then inserts into the representative table.
    4. Formats the topk reps and dists and inserts into the topk table.
    Notice: we assume the data in blob table has same order as its embedding in vector database
    """
    if self.tasti_config is None:
      raise Exception('TASTI hasn\'t been initialized, please provide TASTI config')
    tasti_index = Tasti(self.tasti_config)
    rep_ids = tasti_index.get_representative_blob_ids()
    topk_for_all = tasti_index.vector_database.get_topk_representatives_for_all()
    topk = max(topk_for_all['topk_reps'].str.len())

    await self._create_tasti_table(topk)
    #FIXME: if user provide blob id table, directly read from that table
    blob_key_query = self._get_blob_key_query()
    blob_key_data = await self.execute_query(blob_key_query)
    selected_blob_key_data = blob_key_data.iloc[rep_ids['blob_id'].values.tolist()]
    rep_data = pd.concat([rep_ids.reset_index(drop=True), selected_blob_key_data.reset_index(drop=True)], axis=1)

    await self.insert_data(rep_data, self.reps_table_name)

    new_topk_for_all = self._format_topk_for_all(topk_for_all)
    await self.insert_data(new_topk_for_all, self.topk_table_name)

  #TODO: update topk table and representative table, query for new embeddings.