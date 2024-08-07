import math
import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from typing import List

from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query
from aidb.utils.constants import MASS_COL_NAME, PROXY_SCORE_COL_NAME, SEED_PARAMETER, VECTOR_ID_COLUMN, WEIGHT_COL_NAME
from aidb.utils.logger import logger

BUDGET = 10000


class ApproxSelectEngine(TastiEngine):
  async def get_inference_results(self, query:Query, sampled_index: List[int], conn):
    # TODO: rewrite query, use full scan to execute query
    bound_service_list = query.inference_engines_required_for_query
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filtered_index(bound_service,
                                                                                self.blob_mapping_table_name,
                                                                                sampled_index)
      inp_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(inp_query_str), conn))
      inp_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
      await bound_service.infer(inp_df)


    query_no_aqp = query.base_sql_no_aqp
    query_after_normalize = query_no_aqp.query_after_normalizing
    query_after_adding_vector_id_column = query_after_normalize.add_select(
        f'{self.blob_mapping_table_name}.{VECTOR_ID_COLUMN}'
    )

    blob_keys = [col.name for col in self._config.tables[self.blob_mapping_table_name].columns]
    added_cols = set()
    join_conditions = []
    tables_in_query = query_no_aqp.tables_in_query
    for table_name in tables_in_query:
      for col in self._config.tables[table_name].columns:
        col_name = col.name
        if col_name in blob_keys and col_name not in added_cols:
          join_conditions.append(f'{self.blob_mapping_table_name}.{col_name} = {table_name}.{col_name}')
          added_cols.add(col_name)
    join_conditions_str = ' AND '.join(join_conditions)

    query_after_adding_join = query_after_adding_vector_id_column.add_join(
        f'JOIN {self.blob_mapping_table_name} ON {join_conditions_str}'
    )
    
    table_columns = [f'{self.blob_mapping_table_name}.{VECTOR_ID_COLUMN}']
    sampled_df = pd.DataFrame({VECTOR_ID_COLUMN: sampled_index})

    async with self._sql_engine.begin() as conn:
      sample_query_add_filter_key, _ = self.add_filter_key_into_query(
          table_columns,
          sampled_df,
          query_after_adding_join
      )
      res_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(sample_query_add_filter_key.sql_str), conn))
      # We need to add '__vector_id' in SELECT clause. When 'SELECT *', there will be two '__vector_id' columns.
      # So we need to drop duplicated columns
      res_df = res_df.loc[:, ~res_df.columns.duplicated()]
      res_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
      
    return res_df


  def get_sampled_proxy_blob(self, proxy_score_for_all_blobs, defensive_mixing: int = 0.1):
    weights = proxy_score_for_all_blobs.values ** 0.5
    normalized_weights = (1 - defensive_mixing) * (weights / sum(weights)) + defensive_mixing / len(weights)
    mass = 1 / len(weights) / normalized_weights

    dataset = pd.DataFrame({
        PROXY_SCORE_COL_NAME: proxy_score_for_all_blobs.values,
        WEIGHT_COL_NAME: normalized_weights,
        MASS_COL_NAME: mass},
        index=proxy_score_for_all_blobs.index
    )

    return dataset


  def tau_estimate(self, recall_target, satisfied_sampled_results):
    estimated_tau = 0.0
    recall_score = 0

    for index, blob in satisfied_sampled_results.iterrows():
      recall_score += blob[MASS_COL_NAME]
      if recall_score >= recall_target:
        estimated_tau = blob[PROXY_SCORE_COL_NAME]
        break
    return estimated_tau


  def _get_confidence_bounds(self, mu, sigma, s, delta):
    if s == 0:
      return 0.0, 0.0
    val = (sigma / math.sqrt(s)) * math.sqrt(2 * math.log(1 / delta))
    return mu - val, mu + val


  def tau_modified(self, satisfied_sampled_results, recall_target, confidence, total_length):
    z = satisfied_sampled_results[MASS_COL_NAME]
    estimated_tau = self.tau_estimate(
      recall_target * sum(z),
      satisfied_sampled_results
    )
    grouped = satisfied_sampled_results.groupby(level=0)
    aggregated = grouped.agg(sum_mass=(MASS_COL_NAME, 'sum'), __proxy_score=(PROXY_SCORE_COL_NAME, 'mean'))
    samples_above_cutoff = aggregated[aggregated[PROXY_SCORE_COL_NAME] >= estimated_tau]
    samples_below_cutoff = aggregated[aggregated[PROXY_SCORE_COL_NAME] < estimated_tau]

    estimated_z1 = list(samples_above_cutoff['sum_mass']) + [0] * (total_length - len(samples_above_cutoff))
    estimated_z2 = list(samples_below_cutoff['sum_mass']) + [0] * (total_length - len(samples_below_cutoff))
    
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
    logger.info(f'modified_recall: {modified_recall_target}')
    logger.info(f'modified_tau: {modified_tau}')
    return modified_tau


  async def execute_approx_select_query(self, query: Query, **kwargs):
    if not query.is_valid_approx_select_query:
      raise Exception('Approx select query should contain Confidence and should not contain Budget.')

    # generate proxy score for each blob
    proxy_score_for_all_blobs = await self.get_proxy_scores_for_all_blobs(query)

    dataset = self.get_sampled_proxy_blob(proxy_score_for_all_blobs)

    if SEED_PARAMETER in kwargs:
      seed = kwargs[SEED_PARAMETER]
    else:
      seed = None
    sampled_df = dataset.sample(BUDGET, replace=True, weights=WEIGHT_COL_NAME, random_state=seed)

    async with self._sql_engine.begin() as conn:
      satisfied_sampled_results = await self.get_inference_results(
          query,
          list(sampled_df.index),
          conn
      )

      joined_satisfied_sampled_results = satisfied_sampled_results.join(sampled_df, how='inner')
      sorted_satisfied_sampled_results = joined_satisfied_sampled_results.sort_values(by=PROXY_SCORE_COL_NAME, ascending=False)

      recall_target = query.recall_target
      confidence = query.confidence / 100
      tau_modified = self.tau_modified(
          sorted_satisfied_sampled_results,
          recall_target,
          confidence,
          BUDGET
      )

      pilot_sample_index = sorted_satisfied_sampled_results.index
      additional_sample_index = dataset[dataset[PROXY_SCORE_COL_NAME] >= tau_modified].index

      additional_selected_blob = list(set(additional_sample_index) - set(pilot_sample_index))
      logger.info(f'num_samples: {len(additional_selected_blob)}')

      new_satisfied_sampled_results = await self.get_inference_results(
          query,
          additional_selected_blob,
          conn
      )
      all_satisfied_sampled_results = pd.concat([satisfied_sampled_results, new_satisfied_sampled_results], ignore_index=True)

    return all_satisfied_sampled_results
