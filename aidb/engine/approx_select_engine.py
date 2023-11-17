import numpy as np
import math
import pandas as pd
from sqlalchemy.sql import text
from typing import List
import time

from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query
from aidb.utils.constants import VECTOR_ID_COLUMN

PROXY_SCORE = 'proxy_score'
LABEL = 'label'
MASS = 'mass'
WEIGHT = 'weight'
BUDGET = 2000

class ApproxSelectEngine(TastiEngine):
  # TODO: design a better algorithm
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


  async def execute_approx_select_query(self, query: Query):
    # generate proxy score for each blob
    score_for_all_df, score_connected = await self.get_proxy_scores_for_all_blobs(query)

    proxy_score_for_all_blobs = self.score_fn(score_for_all_df, score_connected)

    dataset = self.get_sampled_proxy_blob(proxy_score_for_all_blobs)
    sampled_df = dataset.sample(BUDGET, replace=True, weights=WEIGHT)
    # sorted blob id based on proxy score

    async with self._sql_engine.begin() as conn:
      time1 = time.time()
      sampled_results = await self.get_inference_results(query, list(sampled_df.index), conn)

      true_vector_id = await self.get_sampled_true_vecor_id(sampled_results, list(sampled_df.index), conn)

      sampled_df.loc[true_vector_id, LABEL] = 1
      sorted_sampled_df = sampled_df.sort_values(by=PROXY_SCORE, ascending=False)

      recall_target = query.recall_target
      confidence = query.confidence / 100
      tau_modified = self.tau_modified(recall_target, sorted_sampled_df, confidence)

      R1 = sampled_df[sampled_df[LABEL] == 1].index
      R2 = dataset[dataset[PROXY_SCORE] > tau_modified].index

      print('num_samples', len(list(set(R1).union(set(R2)))))
      results = await self.get_inference_results(query, list(set(R1).union(set(R2))), conn)

    return results


  async def get_inference_results(self, query:Query, sampled_index: List[int], conn):
    bound_service_list = query.inference_engines_required_for_query
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filtered_index(bound_service,
                                                                                self.blob_mapping_table_name,
                                                                                sampled_index)
      inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
      inp_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
      await bound_service.infer(inp_df)

    res_df = await conn.run_sync(lambda conn: pd.read_sql(text(query.base_sql_no_aqp.sql()), conn))
    return res_df


  async def get_sampled_true_vecor_id(self, results_df, sampled_index: List[int], conn):

    blob_mapping_table_cols = [f'{self.blob_mapping_table_name}.{col.name}'
        for col in self._config.tables[self.blob_mapping_table_name].columns]


    select_vector_id_query = f'''SELECT {VECTOR_ID_COLUMN} FROM {self.blob_mapping_table_name};'''
    select_vector_id_query = Query(select_vector_id_query, config=self._config)
    query_expression = select_vector_id_query.get_expression()
    select_vector_id_query, _ = self.add_filter_key_into_query(
        blob_mapping_table_cols,
        results_df,
        select_vector_id_query,
        query_expression
    )

    true_vector_id_df = await conn.run_sync(lambda conn: pd.read_sql(text(select_vector_id_query.sql()), conn))
    true_vector_id_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
    true_vector_id = list(set(sampled_index).intersection(set(true_vector_id_df.index)))

    return true_vector_id


  def get_sampled_proxy_blob(self, proxy_score_for_all_blobs, defensive_mixing: int = 0.1):
    weights = proxy_score_for_all_blobs.values ** 0.5
    normalized_weights = (1 - defensive_mixing) * (weights / sum(weights)) + defensive_mixing / len(weights)
    mass = 1 / len(weights) / normalized_weights
    label = [0] * len(weights)
    dataset = pd.DataFrame({
        PROXY_SCORE: proxy_score_for_all_blobs.values,
        WEIGHT: normalized_weights,
        MASS: mass,
        LABEL: label},
        index=proxy_score_for_all_blobs.index
    )

    return dataset


  def tau_estimate(self, recall_target, sampled_blobs):
    estimated_tau = 0.0
    # TODO: sorted blobs based on proxy score
    recall_score = 0
    count = 0
    for index, blob in sampled_blobs.iterrows():
      count += 1
      recall_score += blob[LABEL] * blob[MASS]
      if recall_score >= recall_target:
        estimated_tau = blob[PROXY_SCORE]
        break
    return estimated_tau, count


  def _get_confidence_bounds(self, mu, sigma, s, delta):
    if s == 0:
      return 0.0, 0.0
    val = (sigma / math.sqrt(s)) * math.sqrt(2 * math.log(1 / delta))
    return mu - val, mu + val


  def tau_modified(self, recall_target, sampled_blobs, confidence):
    z = sampled_blobs[MASS] * sampled_blobs[LABEL]
    estimated_tau, top_index = self.tau_estimate(
      recall_target * sum(z),
      sampled_blobs
    )

    # proxy scores samples are sorted
    z1_mask = np.array([1] * top_index + [0] * (len(sampled_blobs) - top_index))
    z2_mask = 1 - z1_mask
    estimated_z1 = z * z1_mask
    estimated_z2 = z * z2_mask
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

    modified_tau, _ = self.tau_estimate(
      modified_recall_target * sum(z),
      sampled_blobs
    )

    print(modified_tau)
    return modified_tau
