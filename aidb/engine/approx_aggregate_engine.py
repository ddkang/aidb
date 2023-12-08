import pandas as pd
import scipy
from sqlalchemy.sql import text
import sqlglot.expressions as exp
import statsmodels.stats.proportion
from typing import List

from aidb.engine.full_scan_engine import FullScanEngine
from aidb.estimator.estimator import (Estimator, WeightedMeanSetEstimator,
                                      WeightedCountSetEstimator, WeightedSumSetEstimator)
from aidb.samplers.sampler import SampledBlob
from aidb.query.query import Query
from aidb.utils.logger import logger

_NUM_PILOT_SAMPLES = 1000


class ApproximateAggregateEngine(FullScanEngine):
  async def execute_aggregate_query(self, query: Query):
    '''
    Execute aggregation query using approximate processing
    '''
    query.is_valid_aqp_query()

    blob_tables = query.blob_tables_required_for_query
    filtering_predicates = query.filtering_predicates
    blob_key_filtering_predicates = []
    inference_engines_required = query.inference_engines_required_for_filtering_predicates
    for filtering_predicate, engine_required in zip(filtering_predicates, inference_engines_required):
      if len(engine_required) == 0:
        blob_key_filtering_predicates.append(filtering_predicate)
    blob_key_filtering_predicates_str = self._get_where_str(blob_key_filtering_predicates)
    column_to_root_column = self._config.columns_to_root_column
    for k, v in column_to_root_column.items():
      blob_key_filtering_predicates_str = blob_key_filtering_predicates_str.replace(k, v)
    blob_key_filtering_predicates_str = f'WHERE {blob_key_filtering_predicates_str}' \
                                         if blob_key_filtering_predicates_str else ''

    async with self._sql_engine.begin() as conn:
      blob_count_query_str = self.get_blob_count_query(blob_tables, blob_key_filtering_predicates_str)
      blob_count_res = await conn.execute(text(blob_count_query_str))
      self.blob_count = blob_count_res.fetchone()[0]
      # run inference on pilot blobs
      sample_results = await self.get_results_on_sampled_data(
          _NUM_PILOT_SAMPLES,
          query,
          blob_tables,
          blob_key_filtering_predicates_str,
          conn
      )

      if len(sample_results) == 0:
        raise Exception(
            '''We found no records that match your predicate in 1000 samples, so we can't guarantee the
            error target. Try running without the error target if you are certain you want to run this query.'''
        )

      aggregation_type_list = query.aggregation_type_list_in_query
      num_aggregations = len(aggregation_type_list)
      alpha = (1. - (query.confidence / 100.)) / num_aggregations
      conf = 1. - alpha

      num_samples = self.get_additional_required_num_samples(query, sample_results, alpha)
      logger.info(f'num_samples: {num_samples}')
      # when there is not enough data samples, directly run full scan engine and get exact result
      if num_samples + _NUM_PILOT_SAMPLES >= self.blob_count:
        query_no_aqp = query.base_sql_no_aqp
        res = await self.execute_full_scan(query_no_aqp)
        return res

      new_sample_results = await self.get_results_on_sampled_data(
          num_samples,
          query,
          blob_tables,
          blob_key_filtering_predicates_str,
          conn
      )

    sample_results.extend(new_sample_results)
    # TODO:  figure out what should parameter num_samples be for COUNT/SUM query

    estimates = []

    for index, agg_type in enumerate(aggregation_type_list):
      extracted_sample_results = [sample_result.get_specific_data_by_index(index)
                                      for sample_result in sample_results]

      estimator = self._get_estimator(agg_type)
      estimates.append(
          estimator.estimate(
              extracted_sample_results,
              _NUM_PILOT_SAMPLES + num_samples,
              conf
          ).estimate
      )

    # For approximate aggregation, we currently do not support the GROUP BY clause, so there is only one row result.
    # We still return the result a list of tuple to maintain the format
    return [tuple(estimates)]


  def get_blob_count_query(self, table_names: List[str], blob_key_filtering_predicates_str: str):
    '''Function that returns a query, to get total number of blob ids'''
    join_str = self._get_inner_join_query(table_names)
    return f'''
            SELECT COUNT(*)
            {join_str}
            {blob_key_filtering_predicates_str};
            '''


  def _get_estimator(self, agg_type: exp.Expression) -> Estimator:
    if agg_type == exp.Avg:
      return WeightedMeanSetEstimator(self.blob_count)
    elif agg_type == exp.Sum:
      return WeightedSumSetEstimator(self.blob_count)
    elif agg_type == exp.Count:
      return WeightedCountSetEstimator(self.blob_count)
    else:
      raise NotImplementedError('We only support AVG, COUNT, and SUM for approximate aggregations.')


  def get_sample_blobs_query(self, blob_tables: List[str], num_samples: int, blob_key_filtering_predicates_str: str):
    '''Function to return a query to get all blob keys'''
    join_str = self._get_inner_join_query(blob_tables)

    select_column = []
    col_name_set = set()
    for table in blob_tables:
      for blob_key in self._config.blob_keys[table]:
        col_name = blob_key.split('.')[1]
        if col_name not in col_name_set:
          select_column.append(blob_key)
          col_name_set.add(col_name)

    select_column_str = ', '.join(select_column)
    # FIXME: add condition that samples are not in previous sampled data
    sample_blobs_query_str = f'''
                              SELECT {select_column_str}
                              {join_str}
                              {blob_key_filtering_predicates_str}
                              ORDER BY RANDOM()
                              LIMIT {num_samples};
                              '''
    return sample_blobs_query_str


  async def execute_inference_services(self, query: Query, sample_df: pd.DataFrame, conn):
    '''
    Executed inference services on sampled data
    '''
    bound_service_list = query.inference_engines_required_for_query
    inference_services_executed = set()
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filter_service(
          bound_service,
          query,
          inference_services_executed
      )
      new_query = Query(inp_query_str, self._config)
      query_add_filter_key, _ = self.add_filter_key_into_query(
          list(bound_service.binding.input_columns),
          sample_df,
          new_query
      )

      input_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_add_filter_key.sql_str), conn))
      await bound_service.infer(input_df)
      inference_services_executed.add(bound_service.service.name)


  async def get_results_for_each_blob(self, query: Query, sample_df: pd.DataFrame, conn) -> List[SampledBlob]:
    '''
    Return aggregation results of each sample blob, contains weight, mass, statistics, num_items
    For example, if there is 1000 blobs, blob A has two detected objects with x_min = 500 and x_min = 1000,
    the query is Avg(x_min), the result of blob A is SampledBlob(1/1000, 1, 750, 2)
    mass is default value 1, weight is same for each blob in uniform sampling
    '''
    tables_in_query = query.tables_in_query
    query_no_aqp = query.base_sql_no_aqp

    table_columns = [f'{table}.{col.name}' for table in tables_in_query for col in self._config.tables[table].columns]
    query_add_filter_key, selected_column = self.add_filter_key_into_query(
        table_columns,
        sample_df,
        query_no_aqp
    )
    query_add_count = query_add_filter_key.add_select('COUNT(*) AS num_items')
    query_str = f'''
                  {query_add_count.sql_str}
                  GROUP BY {', '.join(selected_column)}
                 '''

    res_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str), conn))

    results_for_each_blob = []
    for _, row in res_df.iterrows():
      results_for_each_blob.append(SampledBlob(
          weight=1. / self.blob_count,
          mass=1,
          statistics=list(row[:-1]),
          num_items=int(row[-1])
      ))

    return results_for_each_blob


  async def get_results_on_sampled_data(
      self,
      num_samples: int,
      query: Query,
      blob_tables: List[str],
      blob_key_filtering_predicates_str: str,
      conn
  ):
    sample_blobs_query_str = self.get_sample_blobs_query(blob_tables, num_samples, blob_key_filtering_predicates_str)
    sample_blobs_df = await conn.run_sync(lambda conn: pd.read_sql(text(sample_blobs_query_str), conn))

    if len(sample_blobs_df) < num_samples:
      raise Exception(f'Require {num_samples} samples, but only get {len(sample_blobs_df)} samples')

    await self.execute_inference_services(query, sample_blobs_df, conn)

    results_for_each_blob = await self.get_results_for_each_blob(query, sample_blobs_df, conn)

    return results_for_each_blob


  def get_additional_required_num_samples(
      self,
      query: Query,
      sample_results: List[SampledBlob],
      alpha
  ) -> int:
    error_target = query.error_target
    conf = 1. - alpha
    num_samples = []

    for index, agg_type in enumerate(query.aggregation_type_list_in_query):
      estimator = self._get_estimator(agg_type)
      extracted_sample_results = [sample_result.get_specific_data_by_index(index)
                                      for sample_result in sample_results]
      pilot_estimate = estimator.estimate(extracted_sample_results, _NUM_PILOT_SAMPLES, conf)

      if agg_type == exp.Avg:
        p_lb = statsmodels.stats.proportion.proportion_confint(
          len(sample_results),
          _NUM_PILOT_SAMPLES,
          alpha
        )[0]
      else:
        p_lb = 1

      num_samples.append(int(
        (scipy.stats.norm.ppf(alpha / 2) * pilot_estimate.std_ub / (error_target * pilot_estimate.lower_bound)) ** 2 * \
        (1. / p_lb)
      ))

    return max(max(num_samples), 100)
