import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
from typing import Optional

from aidb.config.config_types import Table
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.query.query import Query
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

    self.reps_table_name = None
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

    self.reps_table_name, self.topk_table_name = table_name_for_rep_and_topk(blob_tables)
    reps_df = await self._get_cluster_rep_blob_ids()
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service(bound_service, query,
                                                                 self.reps_table_name, list(reps_df.index))
      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
      inp_df.set_index('blob_id', inplace=True, drop=True)
      reps_df = reps_df.loc[inp_df.index]
      await bound_service.infer(inp_df)

    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql_query_text))
    res = res.fetchall()
    true_score = self.score_fn(res)
    proxy_score_for_all_blob_ids = self.get_proxy_score_for_all_blob_ids(true_score)
    # FIXME: decide what to return
    return res, proxy_score_for_all_blob_ids


  def score_fn(self, res_df: pd.DataFrame) -> pd.DataFrame:
    '''
    convert query result to score, return a Dataframe contains score column, the index is blob index
    '''
    raise NotImplementedError


  async def _get_cluster_rep_blob_ids(self) -> pd.DataFrame:
    if self.reps_table_name not in self._config.tables:
      await self.initialize_tasti()

    reps_query_str = f'''
                SELECT {self.reps_table_name}.blob_id
                FROM {self.reps_table_name};
                '''

    async with self._sql_engine.begin() as conn:
      reps_df = await conn.run_sync(lambda conn: pd.read_sql(text(reps_query_str), conn))
    reps_df.set_index('blob_id', inplace=True, drop=True)

    return reps_df


  async def get_proxy_score_for_all_blob_ids(self, true_score: pd.DataFrame, return_binary_score = True) -> pd.Series:

    topk_query_str = f'SELECT * FROM {self.topk_table_name}'
    async with self._sql_engine.begin() as conn:
      topk_df = await conn.run_sync(lambda conn: pd.read_sql(text(topk_query_str), conn))
    topk_df.set_index('blob_id', inplace=True, drop=True)

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
    rep_table = sqlalchemy.schema.Table(self.reps_table_name, metadata,
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

    self._config.tables[self.reps_table_name] = Table(rep_table)
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
      await conn.run_sync(lambda conn: rep_blob_df.to_sql(self.reps_table_name, conn, if_exists='append', index=False))
      await conn.run_sync(lambda conn: new_topk_for_all.to_sql(self.topk_table_name, conn,
                                                               if_exists='append', index=False))

  # TODO: update topk table and representative table, query for new embeddings.
