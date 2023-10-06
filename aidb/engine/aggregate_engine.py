import pandas as pd
from sqlalchemy.sql import text
import sqlglot.expressions as exp
from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query

from aidb.utils.logger import logger

from typing import List
import scipy
import statsmodels.stats.proportion
from aidb.estimator.estimator import (Estimator, WeightedCountSetEstimator,
                                      WeightedMeanSetEstimator,
                                      WeightedSumSetEstimator)
from aidb.samplers.sampler import SampledBlobId, SampledBlob


class ApproximateAggregateEngine(BaseEngine):
  def _get_estimator(self, agg_type: exp.Expression, columns: List[str]) -> Estimator:
    if agg_type == exp.Sum:
      return WeightedSumSetEstimator(self.num_blob_ids)
    elif agg_type == exp.Avg:
      return WeightedMeanSetEstimator(self.num_blob_ids)
    elif agg_type == exp.Count:
      return WeightedCountSetEstimator(self.num_blob_ids)
    else:
      raise NotImplementedError()
  
  def get_random_sampling_query(self,bound_service, query,
                        inference_services_executed):
    inp_query_str = self.get_input_query_for_inference_service(
                          bound_service,
                          query,
                          inference_services_executed
                        )

    return f'''{inp_query_str.replace(';', '')}
                ORDER BY RANDOM()
                LIMIT {self.num_samples_required};'''


  def get_num_blob_ids_query(self, table, column, num_samples=100):
    num_ids_query = f'''
                      SELECT COUNT({column})
                      FROM {table};
                      '''
    return num_ids_query

  def get_population_percent(self):
    return 10

  def search_sample_columns(self, sample, agg_on_column, agg_table):
    column_of_interest = f'{agg_table}.{agg_on_column}'
    if column_of_interest not in sample.columns:
      return False
    if sample[column_of_interest].empty:
      return False
    return True

  async def get_sampled_blobs_with_stats(self, infer_output, agg_on_column, agg_table, query, est_conn):
    sampled_blobs = []
    num_samples = len(infer_output)
    mass = 1.
    wt = 1. / (num_samples)
    for idx in range(num_samples):
      sampled_blob_id = SampledBlobId(
                          infer_output[idx],
                          1. / num_samples,
                          1.
                        )
      blob_table = {agg_table: infer_output[idx]}
      if not self.search_sample_columns(infer_output[idx], agg_on_column, agg_table):
        continue

      map_cols = {}
      for col in infer_output[idx].columns:
        hier_col = col.split('.')
        map_cols[col] = hier_col[1] if len(hier_col) > 1 else hier_col[0]
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
      sampled_blobs.append(SampledBlob(
                              infer_output[idx],
                              1. / num_samples,
                              1.,
                              blob_table,
                              statistic_ans,
                              {agg_table: len(infer_output[idx])}
                          )
      )
    return sampled_blobs


  async def execute_aggregate_query(self, query: Query, dialect=None):
    # Get the base SQL query, columns
    tables_in_query = query.tables_in_query
    columns = query._columns
    agg_type = query.get_agg_type()
    agg_on_column = query.get_aggregated_column(agg_type)

    all_tables_columns = {}
    for table in self._config.tables.values():
      all_tables_columns[table.name] = table.columns
    agg_on_column_table = query.get_table_of_column(agg_on_column, all_tables_columns, tables_in_query)
  
    conf = query.get_confidence() / 100.
    error_target = query.get_error_target()
    population_percent = self.get_population_percent()
    alpha = 1. - conf

    def get_num_ids_query():
      inp_table = self._config.blob_tables[0]
      inp_col = self._config.blob_keys[inp_table][0]
      num_ids_query = self.get_num_blob_ids_query(
                                inp_table,
                                inp_col
                        )
      return num_ids_query
    num_ids_query = get_num_ids_query()
    async with self._sql_engine.begin() as conn:
      self.num_blob_ids = await conn.run_sync(
                            lambda conn: pd.read_sql(
                                text(num_ids_query),
                                conn
                              )
                          )
    self.num_blob_ids = int(self.num_blob_ids.iloc[0, 0])

    #for pilot run
    self.num_samples_required = int(self.num_blob_ids * population_percent / 100.)

    service_ordering = self._config.inference_topological_order
    inference_services_executed = set()
    for bound_service in service_ordering:
      if not f'{agg_on_column_table}.{agg_on_column}' in bound_service.binding.output_columns:
        continue
      out_query = self.get_random_sampling_query(
                          bound_service,
                          query,
                          inference_services_executed
                        )
      async with self._sql_engine.begin() as conn:
        out_df = await conn.run_sync(
                          lambda conn: pd.read_sql(
                            text(out_query),
                            conn
                          )
                        )
      input_samples = await bound_service.infer(out_df)

      async with self._sql_engine.begin() as conn:
        sampled_blobs = await self.get_sampled_blobs_with_stats(input_samples, agg_on_column, agg_on_column_table, query, conn)

      estimator = self._get_estimator(agg_type, columns)
      pilot_estimate = estimator.estimate(sampled_blobs, self.num_samples_required, conf, agg_table=agg_on_column_table)
      p_lb = statsmodels.stats.proportion.proportion_confint(len(sampled_blobs), self.num_samples_required, conf)[0]
      num_samples = int(
        (scipy.stats.norm.ppf(1. - alpha / 2) * pilot_estimate.std_ub / error_target) ** 2 *\
          (1. / p_lb)
          )
      self.num_samples_required = num_samples
      final_query = self.get_random_sampling_query(
                          bound_service,
                          query,
                          inference_services_executed
                        )
      async with self._sql_engine.begin() as conn:
        final_df = await conn.run_sync(
                          lambda conn: pd.read_sql(
                            text(final_query),
                            conn
                          )
                        )
      all_samples = await bound_service.infer(final_df)
      input_samples.extend(all_samples)

      async with self._sql_engine.begin() as conn:
        all_blobs = await self.get_sampled_blobs_with_stats(input_samples, agg_on_column, agg_on_column_table, query, conn)
      return [(estimator.estimate(all_blobs, num_samples, conf, agg_table=agg_on_column_table).estimate,)]