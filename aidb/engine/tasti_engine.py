import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
from typing import Dict, List, Optional

from aidb.config.config_types import Table
from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query
from aidb.utils.constants import table_name_for_rep_and_topk_and_blob_mapping, VECTOR_ID_COLUMN
from aidb.vector_database.tasti import Tasti


class TastiEngine(BaseEngine):
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      debug: bool = False,
      tasti_index: Optional[Tasti] = None,
      user_specified_vector_ids: Optional[pd.DataFrame] = None
  ):
    super().__init__(connection_uri, infer_config, debug)

    self.rep_table_name = None
    # TODO: modify to same rep table with different topk table
    self.topk_table_name = None
    # table for mapping blob keys to blob ids
    self.blob_mapping_table_name = None
    self.tasti_index = tasti_index
    self.user_specified_vector_ids = user_specified_vector_ids


  async def get_proxy_scores_for_all_blobs(self, query: Query, return_binary_score= False, **kwargs):
    '''
    1. create rep table and topk table if not exist, store the results from vector database
    2. infer all bound services for all cluster representatives blobs
    3. generate proxy score per predicate for all blobs based on topk rep ids and dists
    '''
    bound_service_list = query.inference_engines_required_for_query
    blob_tables = query.blob_tables_required_for_query

    self.rep_table_name, self.topk_table_name, self.blob_mapping_table_name = \
        table_name_for_rep_and_topk_and_blob_mapping(blob_tables)

    if self.rep_table_name not in self._config.tables or self.topk_table_name not in self._config.tables:
      await self.initialize_tasti()

    async with self._sql_engine.begin() as conn:
      reps_query_str = f'SELECT * FROM {self.rep_table_name}'
      reps_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(reps_query_str), conn))

    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filtered_index(bound_service, self.rep_table_name)
      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(inp_query_str), conn))
      inp_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
      await bound_service.infer(inp_df)

    score_query_str = self.get_score_query_str(query, self.rep_table_name)
    async with self._sql_engine.begin() as conn:
      score_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(score_query_str), conn))
    score_df = pd.merge(reps_df, score_df, on=VECTOR_ID_COLUMN, how='left')
    score_df.fillna(0, inplace=True)
    score_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
    score_df = score_df['score']
    score_for_all_df = await self.propagate_score_for_all_vector_ids(score_df, return_binary_score)

    # FIXME: decide what to return for different usage: Limit engine, Aggregation, Full scan optimize.
    return score_for_all_df


  def get_score_query_str(self, query: Query, rep_table_name: str) -> (str, List[List[str]]):
    '''
    Convert filtering condition into select clause, so we can use select result to compute proxy score for all blobs
    '''
    where_condition = query.convert_and_connected_fp_to_exp(query.filtering_predicates)
    if where_condition:
      where_str = f'WHERE {where_condition.sql()}'
    else:
      where_str = ''
    filtering_predicates = [fp.sql() for and_connected in query.filtering_predicates for fp in and_connected]
    tables = self._get_tables(filtering_predicates)

    # some representative blobs may not have outputs from service, so left join is better
    join_str = self._get_left_join_str(rep_table_name, tables)
    score_query_str = f'''
                      SELECT {rep_table_name}.{VECTOR_ID_COLUMN}, COUNT(*) AS score
                      {join_str}
                      {where_str}
                      GROUP BY {rep_table_name}.{VECTOR_ID_COLUMN};
                      '''
    return score_query_str


  async def propagate_score_for_all_vector_ids(self, score_df: pd.Series, return_binary_score = False) -> pd.Series:
    topk_query_str = f'SELECT * FROM {self.topk_table_name}'
    async with self._sql_engine.begin() as conn:
      topk_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(topk_query_str), conn))
    topk_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)

    topk = len(topk_df.columns) // 2
    topk_indices = np.arange(topk)
    all_dists = np.zeros((len(topk_df), topk))
    all_scores = np.zeros((len(topk_df), topk))

    for idx, (index, row) in enumerate(topk_df.iterrows()):
      reps = [int(row[f'topk_reps_{i}']) for i in topk_indices]
      dists = [row[f'topk_dists_{i}'] for i in topk_indices]

      if index in score_df.index:
        reps = [index] * topk
        dists = [1] * topk

      all_dists[idx, :] = dists
      all_scores[idx, :] = score_df.loc[reps].values
    # to avoid division by zero error
    all_dists += 1e-8

    if return_binary_score:
      all_scores = np.where(all_scores == 0, 0, 1)
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


  def _create_tasti_table(self, topk:int,  conn: sqlalchemy.engine.base.Connection):
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
    columns.append(sqlalchemy.Column(VECTOR_ID_COLUMN, sqlalchemy.Integer, primary_key=True, autoincrement=False))
    for i in range(topk):
      columns.append(sqlalchemy.Column(f'topk_reps_{str(i)}', sqlalchemy.Integer))
      columns.append(sqlalchemy.Column(f'topk_dists_{str(i)}', sqlalchemy.Float(32)))
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
    new_topk_for_all.update({
      VECTOR_ID_COLUMN: [vector_id for vector_id in topk_for_all.index]
    })
    return pd.DataFrame(new_topk_for_all, index=topk_for_all.index)


  async def initialize_tasti(self):
    """
    This function initializes the Tasti index and manages the insertion of data into
    representative and topk tables.
    It performs the following main steps:
    1. Initializes a Tasti index and retrieves representative blob ids and topk reps and dists.
    2. Creates Tasti tables and retrieves culster representative blob key columns.
    3. Inserts data into the representative table and topk table
    """
    if self.tasti_index is None:
      raise Exception('TASTI hasn\'t been initialized, please provide tasti_index')

    if self.user_specified_vector_ids is not None:
      vector_ids = self.user_specified_vector_ids
    else:
      vector_id_select_query_str = f'''
                                    SELECT {self.blob_mapping_table_name}.{VECTOR_ID_COLUMN}
                                    FROM {self.blob_mapping_table_name};
                                    '''

      async with self._sql_engine.begin() as conn:
        vector_ids = await conn.run_sync(lambda conn: pd.read_sql_query(text(vector_id_select_query_str), conn))
    self.tasti_index.set_vector_ids(vector_ids)

    rep_ids = self.tasti_index.get_representative_vector_ids()
    topk_for_all = self.tasti_index.get_topk_representatives_for_all()
    new_topk_for_all = self._format_topk_for_all(topk_for_all)
    topk = max(topk_for_all['topk_reps'].str.len())

    rep_blob_query_str = f'''
                          SELECT *
                          FROM {self.blob_mapping_table_name}
                          WHERE {self.blob_mapping_table_name}.{VECTOR_ID_COLUMN} IN {format(tuple(rep_ids.index))};
                          '''

    async with self._sql_engine.begin() as conn:
      await conn.run_sync(lambda conn: self._create_tasti_table(topk, conn))
      rep_blob_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(rep_blob_query_str), conn))

      await conn.run_sync(lambda conn: rep_blob_df.to_sql(self.rep_table_name, conn, if_exists='append', index=False))
      await conn.run_sync(lambda conn: new_topk_for_all.to_sql(self.topk_table_name, conn,
                                                               if_exists='append', index=False))

  # TODO: update topk table and representative table, query for new embeddings.
