from collections import defaultdict
from typing import List

import pandas as pd
import scipy
import sqlglot.expressions as exp
import statsmodels.stats.proportion
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.estimator.estimator import (Estimator, WeightedCountSetEstimator,
                                      WeightedMeanSetEstimator,
                                      WeightedSumSetEstimator)
from aidb.query.query import Query
from aidb.samplers.sampler import SampledBlob
from aidb.utils.constants import NUM_PILOT_SAMPLES
from aidb.utils.logger import logger


class ApproximateAggregateEngine(BaseEngine):
  def _get_estimator(self, agg_type: exp.Expression) -> Estimator:
    if agg_type == exp.Sum:
      return WeightedSumSetEstimator(self.num_blob_ids)
    elif agg_type == exp.Avg:
      return WeightedMeanSetEstimator(self.num_blob_ids)
    elif agg_type == exp.Count:
      return WeightedCountSetEstimator(self.num_blob_ids)
    else:
      raise NotImplementedError()


  def get_random_sampling_query(self, bound_service, query,
                                inference_services_executed, num_samples):
    '''Function to return limited number of samples randomly'''
    _, inp_cols_str, join_str, where_str = self.get_input_query_for_inference_service(
                                              bound_service,
                                              query,
                                              inference_services_executed
                                          )
    if where_str:
      return f'''
                SELECT {inp_cols_str}
                {join_str}
                WHERE {where_str}
                ORDER BY RANDOM()
                LIMIT {num_samples};
              '''

    else:
      return f'''
                SELECT {inp_cols_str}
                {join_str}
                ORDER BY RANDOM()
                LIMIT {num_samples};
              '''


  def get_num_blob_ids_query(self, table):
    '''Function that returns a query, to get total number of blob ids'''
    num_ids_query = f'''
                    SELECT COUNT(*)
                    FROM {table};
                    '''
    return num_ids_query


  def search_sample_columns(self, sample, agg_on_columns):
    '''
    Function to search if our aggregated columns of interest 
    are in corresponding sample taken, so that blob can be sampled.
    '''
    if any(column not in sample.columns for column in agg_on_columns):
      return False
    if all(sample[column].empty for column in agg_on_columns):
      return False
    return True


  def find_num_required_samples(self, estimator, sampled_blobs, conf, alpha, error_target,
                                  agg_on_column_table):
    pilot_estimate = estimator.estimate(
        sampled_blobs,
        NUM_PILOT_SAMPLES,
        conf / 2,
        True,
        agg_table=agg_on_column_table
    )
    p_lb = statsmodels.stats.proportion.proportion_confint(
        len(sampled_blobs),
        NUM_PILOT_SAMPLES,
        alpha / 2.)[0]
    num_samples = int(
        (scipy.stats.norm.ppf(alpha / 2) *
         pilot_estimate.std_ub / error_target) ** 2 *
        (1. / p_lb)
    )
    return num_samples


  async def get_samples_of_concerned_services(self, inference_services_required, query, conn,
                                          num_samples, inference_services_executed = set()):
    samples_concerned = defaultdict(lambda: None)
    for bound_service in self._config.inference_topological_order:
      if bound_service not in inference_services_required:
        continue
      out_query = self.get_random_sampling_query(
          bound_service,
          query,
          inference_services_executed,
          num_samples
      )
      async with self._sql_engine.begin() as conn:
        out_df = await conn.run_sync(
            lambda conn: pd.read_sql(
                text(out_query),
                conn
            )
        )
      samples_concerned[bound_service.service.name] = await bound_service.infer(out_df)
      inference_services_executed.add(bound_service.service.name)
    samples = []
    for inference_service in inference_services_required:
      samples.extend(samples_concerned[inference_service.service.name])
    return samples, inference_services_executed


  async def get_sampled_blobs_with_stats(self, infer_output, agg_on_columns,
                                         agg_table, query, est_conn):
    '''
    Given inference output as input, this function,
    iterates through the list of dataframes,
    executes query on them, computes statistic on relevant rows,
    returns SampledBlobs that can be passes to estimator
    '''
    sampled_blobs = []
    num_samples = len(infer_output)
    mass = 1.
    wt = 1. / (num_samples)
    for idx in range(num_samples):
      blob_table = {agg_table: infer_output[idx]}
      if not self.search_sample_columns(infer_output[idx], agg_on_columns):
        continue

      map_cols = {}
      for col in infer_output[idx].columns:
        table_column = col.split('.')
        map_cols[col] = table_column[1] if len(
            table_column) > 1 else table_column[0]
      infer_output[idx].rename(columns=map_cols, inplace=True)
      proxy_table = f'{agg_table}_proxy'

      await est_conn.run_sync(
          lambda sync_conn: infer_output[idx].to_sql(
              proxy_table,
              con=sync_conn,
              if_exists='replace',
              index=False
          )
      )
      new_query = query.sql_query_text.replace(agg_table, proxy_table)
      res = await est_conn.execute(text(new_query))
      res = res.fetchone()
      statistic_ans = res[0]
      if statistic_ans is None:
        continue
      sampled_blobs.append(
          SampledBlob(
              infer_output[idx],
              wt,
              mass,
              blob_table,
              statistic_ans,
              {agg_table: len(infer_output[idx])}
          )
      )
    return sampled_blobs


  async def execute_aggregate_query(self, query: Query, dialect=None):
    '''
    Execute aggregation query using approximate processing
    Supports single aggregations for now, 
    to be extended to support multi-aggregations
    '''
    tables_in_query = query.tables_in_query
    agg_type = query.get_agg_type()
    agg_on_columns = query.get_aggregated_columns(agg_type)
    agg_on_column_table = query.get_table_of_column(
                            agg_on_columns[0].split('.')[1]) # support single aggregation queries 
    inference_services_required = set()
    for col in agg_on_columns:
      inference_services_required.add(query.config.column_by_service[col])
    error_target = query.error_target
    conf = query.confidence / 100.
    alpha = 1. - conf

    async with self._sql_engine.begin() as conn:
      # first blob table is only considered for now
      self.num_blob_ids = await conn.execute(text(
                            self.get_num_blob_ids_query(self._config.blob_tables[0])))
    self.num_blob_ids = self.num_blob_ids.fetchone()[0]

    # Pilot run and inference to determine required number of samples
    input_samples, inference_services_executed = await self.get_samples_of_concerned_services(
                                                        inference_services_required,
                                                        query,
                                                        conn,
                                                        NUM_PILOT_SAMPLES
                                                )
    async with self._sql_engine.begin() as conn:
      sampled_blobs = await self.get_sampled_blobs_with_stats(
                        input_samples, agg_on_columns, agg_on_column_table, query, conn)
    estimator = self._get_estimator(agg_type)
    num_samples = self.find_num_required_samples(estimator, sampled_blobs,
                        conf, alpha, error_target, agg_on_column_table)
    if not num_samples:
      logger.debug(f'Approx estimate is zero, going for full sampling')
      num_samples = self.num_blob_ids

    
    # Final query execution on required number of samples
    all_samples, _ = await self.get_samples_of_concerned_services(
                            inference_services_required,
                            query,
                            conn,
                            num_samples,
                            inference_services_executed
                        )
    async with self._sql_engine.begin() as conn:
      all_blobs = await self.get_sampled_blobs_with_stats(
                          all_samples, agg_on_columns, agg_on_column_table, query, conn)
    all_samples_estimate = estimator.estimate(
        all_blobs,
        num_samples,
        conf,
        False,
        agg_table=agg_on_column_table
    )
    return [(all_samples_estimate.estimate,)]