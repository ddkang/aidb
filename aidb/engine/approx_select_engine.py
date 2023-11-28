import math
import multiprocessing as mp
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from typing import List

from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query
from aidb.utils.constants import VECTOR_ID_COLUMN
from aidb.utils.logger import logger

BUDGET = 10000
MASS = 'mass'
PROXY_SCORE = 'proxy_score'
SEED = 'seed'
WEIGHT = 'weight'

class ApproxSelectEngine(TastiEngine):
  # TODO: design a better algorithm, this function is same as the function in Limit Engine
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


  async def get_inference_results(self, query:Query, sampled_index: List[int], conn):
    # TODO: rewrite query, use full scan to execute query
    bound_service_list = query.inference_engines_required_for_query
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filtered_index(bound_service,
                                                                                self.blob_mapping_table_name,
                                                                                sampled_index)
      inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
      inp_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
      await bound_service.infer(inp_df)


    no_aqp_query = Query(query.base_sql_no_aqp.sql(), self._config)
    new_exp = no_aqp_query.add_select(
        no_aqp_query.expression_after_normalize_columns,
        f'{self.blob_mapping_table_name}.{VECTOR_ID_COLUMN}'
    )

    blob_keys = [col.name for col in self._config.tables[self.blob_mapping_table_name].columns]
    added_cols = set()
    join_conditions = []
    tables_in_query = no_aqp_query.tables_in_query
    table_alias = no_aqp_query.table_name_to_aliases
    for table_name in tables_in_query:
      for col in self._config.tables[table_name].columns:
        col_name = col.name
        if col_name in blob_keys and col_name not in added_cols:
          if table_name in table_alias:
            table_name = table_alias[table_name]
          join_conditions.append(f'{self.blob_mapping_table_name}.{col_name} = {table_name}.{col_name}')
          added_cols.add(col_name)
    join_conditions_str = ' AND '.join(join_conditions)

    new_exp = no_aqp_query.add_join(new_exp, f'JOIN {self.blob_mapping_table_name} ON {join_conditions_str}')
    new_query = Query(new_exp.sql(), self._config)

    table_columns = [f'{self.blob_mapping_table_name}.{VECTOR_ID_COLUMN}']
    sampled_df = pd.DataFrame({VECTOR_ID_COLUMN: sampled_index})

    all_query_exp, _ = self.add_filter_key_into_query(
        table_columns,
        sampled_df,
        new_query,
        new_query.base_sql_no_where
    )

    all_df = await conn.run_sync(lambda conn: pd.read_sql(text(all_query_exp.sql()), conn))
    # drop duplicated columns, this will happen when 'select *'
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]
    all_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)

    sample_query_exp, _ = self.add_filter_key_into_query(
        table_columns,
        sampled_df,
        new_query,
        new_query.get_expression()
    )

    res_df = await conn.run_sync(lambda conn: pd.read_sql(text(sample_query_exp.sql()), conn))
    # We need to add '__vector_id' in SELECT clause. When 'SELECT *', there will be two '__vector_id' columns.
    # So we need to drop duplicated columns
    res_df = res_df.loc[:, ~res_df.columns.duplicated()]
    res_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)

    return res_df, all_df


  def get_sampled_proxy_blob(self, proxy_score_for_all_blobs, defensive_mixing: int = 0.1):
    weights = proxy_score_for_all_blobs.values ** 0.5
    normalized_weights = (1 - defensive_mixing) * (weights / sum(weights)) + defensive_mixing / len(weights)
    mass = 1 / len(weights) / normalized_weights

    dataset = pd.DataFrame({
        PROXY_SCORE: proxy_score_for_all_blobs.values,
        WEIGHT: normalized_weights,
        MASS: mass},
        index=proxy_score_for_all_blobs.index
    )

    return dataset


  def tau_estimate(self, recall_target, satisfied_sampled_results):
    estimated_tau = 0.0
    recall_score = 0

    for index, blob in satisfied_sampled_results.iterrows():
      recall_score += blob[MASS]
      if recall_score >= recall_target:
        estimated_tau = blob[PROXY_SCORE]
        break
    return estimated_tau


  def _get_confidence_bounds(self, mu, sigma, s, delta):
    if s == 0:
      return 0.0, 0.0
    val = (sigma / math.sqrt(s)) * math.sqrt(2 * math.log(1 / delta))
    return mu - val, mu + val


  def tau_modified(self, satisfied_sampled_results, recall_target, confidence, total_length):
    z = satisfied_sampled_results[MASS]
    estimated_tau = self.tau_estimate(
      recall_target * sum(z),
      satisfied_sampled_results
    )
    positive_length = len(satisfied_sampled_results[satisfied_sampled_results[PROXY_SCORE] >= estimated_tau])

    estimated_z1 = list(z[:positive_length]) + [0] * (total_length - len(z[:positive_length]))
    estimated_z2 = list(z[positive_length:]) + [0] * (total_length - len(z[positive_length:]))
    z1_mean, z1_std = np.mean(estimated_z1), np.std(estimated_z1)
    z2_mean, z2_std = np.mean(estimated_z2), np.std(estimated_z2)

    delta = 1.0 - confidence

    _, ub = self._get_confidence_bounds(z1_mean, z1_std, len(estimated_z1), delta / 2)
    lb, _ = self._get_confidence_bounds(z2_mean, z2_std, len(estimated_z2), delta / 2)

    # inflate recall to correct the sampling errors
    if (ub+lb) == 0.0:
      modified_recall_target = 1.0
    else:
      modified_recall_target = ub / (ub + lb)

    modified_tau = self.tau_estimate(
      modified_recall_target * sum(z),
      satisfied_sampled_results
    )

    logger.info(f'modified_tau: {modified_tau}')
    return modified_tau


  async def execute_approx_select_query(self, query: Query, **kwargs):
    if not query.is_valid_approx_select_query:
      raise Exception('Approx select query should contain Confidence and should not contain Budget.')

    # generate proxy score for each blob
    score_for_all_df, score_connected = await self.get_proxy_scores_for_all_blobs(query)

    proxy_score_for_all_blobs = self.score_fn(score_for_all_df, score_connected)

    dataset = self.get_sampled_proxy_blob(proxy_score_for_all_blobs)

    if SEED in kwargs:
      seed = kwargs[SEED]
    else:
      seed = None
    sampled_df = dataset.sample(BUDGET, replace=True, weights=WEIGHT, random_state=seed)

    async with self._sql_engine.begin() as conn:
      satisfied_sampled_results, all_sampled_results = await self.get_inference_results(
          query,
          list(sampled_df.index),
          conn
      )

      satisfied_sampled_results = satisfied_sampled_results.join(sampled_df, how='inner')
      sorted_satisfied_sampled_results = satisfied_sampled_results.sort_values(by=PROXY_SCORE, ascending=False)

      no_result_length = BUDGET - len(sampled_df.loc[list(set(all_sampled_results.index))])

      all_sampled_results = all_sampled_results.join(sampled_df, how='inner')

      total_length = no_result_length + len(all_sampled_results)

      recall_target = query.recall_target
      confidence = query.confidence / 100
      tau_modified = self.tau_modified(
          sorted_satisfied_sampled_results,
          recall_target,
          confidence,
          total_length
      )

      pilot_sample_index = sorted_satisfied_sampled_results.index
      additional_sample_index = dataset[dataset[PROXY_SCORE] >= tau_modified].index

      all_selected_blob = list(set(pilot_sample_index).union(set(additional_sample_index)))

      logger.info(f'num_samples: {len(all_selected_blob)}')

      all_satisfied_sampled_results, _ = await self.get_inference_results(
          query,
          all_selected_blob,
          conn
      )

    return all_satisfied_sampled_results
