import pandas as pd
import scipy
from sqlalchemy.sql import text
import sqlglot.expressions as exp
import statsmodels.stats.proportion
from typing import List

from aidb.engine.full_scan_engine import FullScanEngine
from aidb.estimator.estimator import (Estimator, WeightedMeanSetEstimator,
                                      WeightedCountSetEstimator, WeightedSumSetEstimator)
from aidb.query.query import Query
from aidb.utils.constants import MASS_COL_NAME, NUM_ITEMS_COL_NAME, WEIGHT_COL_NAME
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

    concatenated_results = pd.concat([sample_results, new_sample_results], ignore_index=True)
    # TODO:  figure out what should parameter num_samples be for COUNT/SUM query

    estimates = []
    fixed_cols = concatenated_results[[NUM_ITEMS_COL_NAME, WEIGHT_COL_NAME, MASS_COL_NAME]]
    for index, (agg_type, _) in enumerate(aggregation_type_list):
      selected_index_col = concatenated_results.iloc[:, [index]]
      extracted_sample_results = pd.concat([selected_index_col, fixed_cols], axis=1)

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
    if self._config.dialect == 'mysql':
      random_func = 'RAND()'
    else:
      random_func = 'RANDOM()'
    # FIXME: add condition that samples are not in previous sampled data
    sample_blobs_query_str = f'''
                              SELECT {select_column_str}
                              {join_str}
                              {blob_key_filtering_predicates_str}
                              ORDER BY {random_func}
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

      input_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(query_add_filter_key.sql_str), conn))
      await bound_service.infer(input_df)
      inference_services_executed.add(bound_service.service.name)


  async def get_results_for_each_blob(self, query: Query, sample_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Return aggregation results of each sample blob, contains weight, mass, statistics, num_items
    For example, if there is 1000 blobs, blob A has two detected objects with x_min = 500 and x_min = 1000,
    the query is Avg(x_min), the result row of blob A is (750, 2, 1/1000, 1)
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
    query_add_count = query_add_filter_key.add_select(f'COUNT(*) AS {NUM_ITEMS_COL_NAME}')
    query_str = f'''
                  {query_add_count.sql_str}
                  GROUP BY {', '.join(selected_column)}
                 '''
    # In MySQL database, after running inference, the database will lose connection. We need to restart the engine,
    async with self._sql_engine.begin() as conn:
      res_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(query_str), conn))
    res_df[WEIGHT_COL_NAME] = 1. / self.blob_count
    res_df[MASS_COL_NAME] = 1

    return res_df


  async def get_results_on_sampled_data(
      self,
      num_samples: int,
      query: Query,
      blob_tables: List[str],
      blob_key_filtering_predicates_str: str,
      conn
  ):
    sample_blobs_query_str = self.get_sample_blobs_query(blob_tables, num_samples, blob_key_filtering_predicates_str)
    sample_blobs_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(sample_blobs_query_str), conn))

    if len(sample_blobs_df) < num_samples:
      raise Exception(f'Require {num_samples} samples, but only get {len(sample_blobs_df)} samples')

    await self.execute_inference_services(query, sample_blobs_df, conn)

    results_for_each_blob = await self.get_results_for_each_blob(query, sample_blobs_df)

    return results_for_each_blob


  def _calculate_required_num_samples(
      self,
      extracted_sample_results: pd.DataFrame,
      sample_size: int,
      agg_type,
      alpha,
      error_target
  ):
    conf = 1 - alpha
    estimator = self._get_estimator(agg_type)
    pilot_estimate = estimator.estimate(extracted_sample_results, sample_size, conf)
    if agg_type == exp.Avg:
      p_lb = statsmodels.stats.proportion.proportion_confint(
        len(extracted_sample_results),
        sample_size,
        alpha
      )[0]
    else:
      p_lb = 1

    return int(
        (scipy.stats.norm.ppf(alpha / 2) * pilot_estimate.std_ub / (error_target * pilot_estimate.lower_bound)) ** 2 * \
        (1. / p_lb)
    )


  def get_additional_required_num_samples(
      self,
      query: Query,
      sample_results: pd.DataFrame,
      alpha
  ) -> int:
    error_target = query.error_target
    num_samples = []

    fixed_cols = sample_results[[NUM_ITEMS_COL_NAME, WEIGHT_COL_NAME, MASS_COL_NAME]]
    for index, (agg_type, _) in enumerate(query.aggregation_type_list_in_query):
      if agg_type == exp.Avg:
        adjusted_error_target = error_target / ( 2 - error_target)
        num_samples.append(
          self._calculate_required_num_samples(
            fixed_cols,
            _NUM_PILOT_SAMPLES,
            exp.Count,
            alpha,
            adjusted_error_target
          )
        )
        
        sample_results['sum_col'] = sample_results.iloc[:, index] * sample_results[NUM_ITEMS_COL_NAME]
        extracted_sample_results = pd.concat([sample_results[['sum_col']], fixed_cols], axis=1)
        num_samples.append(
          self._calculate_required_num_samples(
            extracted_sample_results,
            _NUM_PILOT_SAMPLES,
            exp.Sum,
            alpha,
            adjusted_error_target
          )
        )
        print('num_samples:', num_samples)
      else:
        selected_index_col = sample_results.iloc[:, [index]]
        extracted_sample_results = pd.concat([selected_index_col, fixed_cols], axis=1)
        num_samples.append(
          self._calculate_required_num_samples(
            extracted_sample_results,
            _NUM_PILOT_SAMPLES,
            agg_type,
            alpha,
            error_target
          )
        )

    return max(max(num_samples), 100)
