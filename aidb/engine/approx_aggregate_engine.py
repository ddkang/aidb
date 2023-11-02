import pandas as pd
import scipy
from sqlalchemy.sql import text
import sqlglot.expressions as exp
import statsmodels.stats.proportion
from typing import List

from aidb.engine.base_engine import BaseEngine
from aidb.estimator.estimator import (
    Estimator, WeightedCountSetEstimator, WeightedMeanSetEstimator, WeightedSumSetEstimator)
from aidb.samplers.sampler import SampledBlob, SampledBlobId
from aidb.samplers.uniform_sampler import UniformBlobSampler
from aidb.query.query import Query



_NUM_PILOT_SAMPLES = 1000


class ApproximateAggregateEngine(BaseEngine):
  async def execute_aggregate_query(self, query: Query):
    '''
    Execute aggregation query using approximate processing
    '''
    query.is_valid_aqp_query()

    blob_tables = query.blob_tables_required_for_query
    async with self._sql_engine.begin() as conn:
      blob_count_res = await conn.execute(text(self.get_blob_count_query(blob_tables)))

      all_blobs_query_str = self.get_all_blobs_query(blob_tables)
      all_blobs_df = await conn.run_sync(lambda conn: pd.read_sql(text(all_blobs_query_str), conn))

      self.blob_count = blob_count_res.fetchone()[0]

      sampler = UniformBlobSampler(self.blob_count)

      # run inference on pilot blobs
      sample_results = await self.execute_inference_and_get_results_on_sampled_data(sampler,
                                                                              all_blobs_df,
                                                                              _NUM_PILOT_SAMPLES,
                                                                              query,
                                                                              conn)
      agg_type = query.get_agg_type
      estimator = self._get_estimator(agg_type)
      num_samples = self.get_additional_required_num_samples(query, sample_results, estimator)
      if num_samples == 0:
        return [(estimator.estimate(sample_results, num_samples, query.confidence/ 100.).estimate,)]
      new_sample_results = await self.execute_inference_and_get_results_on_sampled_data(sampler,
                                                                              all_blobs_df,
                                                                              num_samples,
                                                                              query,
                                                                              conn)

    sample_results.extend(new_sample_results)
    return [(estimator.estimate(sample_results, num_samples, query.confidence/ 100.).estimate,)]


  def get_blob_count_query(self, table_names: List[str]):
    '''Function that returns a query, to get total number of blob ids'''
    join_str = self._get_inner_join_query(table_names)
    return f'''
            SELECT COUNT(*)
            {join_str};
          '''


  def _get_estimator(self, agg_type: exp.Expression) -> Estimator:
    if agg_type == exp.Sum:
      return WeightedSumSetEstimator(self.blob_count)
    elif agg_type == exp.Avg:
      return WeightedMeanSetEstimator(self.blob_count)
    elif agg_type == exp.Count:
      return WeightedCountSetEstimator(self.blob_count)
    else:
      raise NotImplementedError("Avg, Count and Sum aggregations are only supported right now")


  def get_all_blobs_query(self, blob_tables: List[str]):
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
    all_blobs_query_str = f'''
                SELECT {select_column_str}
                {join_str}
                 '''
    return all_blobs_query_str


  async def execute_inference_services(self, query: Query, sample_df: pd.DataFrame, conn):
    '''
    Executed inference services on sampled data
    '''
    inference_engines_required = query.inference_engines_required_for_query
    inference_services_executed = set()
    for bound_service in inference_engines_required:
      inp_query_str = self.get_input_query_for_inference_service_filter_service(bound_service, query,
                                                                                inference_services_executed)
      new_query = Query(inp_query_str, self._config)
      query_str, _ = self.add_filter_key_into_query(bound_service.binding.input_columns,
                                                    sample_df,
                                                    new_query,
                                                    new_query._expression)

      input_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str.sql()), conn))
      await bound_service.infer(input_df)
      inference_services_executed.add(bound_service.service.name)


  async def get_agg_results(self, query: Query, sample_df: pd.DataFrame, conn):
    '''
    Run aggregation query group by blob keys, return aggregation value for each blob
    '''
    tables_in_query = query.tables_in_query
    query_no_aqp_sql = query.base_sql_no_aqp
    query_no_aqp_sql = Query(query_no_aqp_sql.sql(), self._config)
    query_expression = query_no_aqp_sql._expression

    table_columns = [f'{table}.{col.name}' for table in tables_in_query for col in self._config.tables[table].columns]
    query_expression, selected_column = self.add_filter_key_into_query(table_columns,
                                                                       sample_df,
                                                                       query_no_aqp_sql,
                                                                       query_expression,
                                                                       add_select=True)

    query_expression = query_no_aqp_sql.add_select(query_expression, 'COUNT(*) AS num_items')
    query_str = f'''
                  {query_expression.sql()}
                  GROUP BY {', '.join(selected_column)}
                 '''

    res_df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str), conn))
    return res_df


  def get_sample_results(
      self,
      sample_blob_key_df: pd.DataFrame,
      samples: List[SampledBlobId],
      agg_results: pd.DataFrame
  ) -> List[SampledBlob]:
    '''
    Return aggregation results of each blob, contains blob_id, weight, mass, statistic, num_items
    '''
    blob_key_sample_mapping = dict()
    for (index, row), sample in zip(sample_blob_key_df.iterrows(), samples):
      blob_key = ', '.join([str(x) for x in row.values])
      blob_key_sample_mapping[blob_key] = sample

    sample_results = []
    for index, row in agg_results[sample_blob_key_df.columns].iterrows():
      blob_key = ', '.join([str(x) for x in row.values])
      sample_results.append(SampledBlob(blob_key_sample_mapping[blob_key].blob_id,
                                        blob_key_sample_mapping[blob_key].weight,
                                        blob_key_sample_mapping[blob_key].mass,
                                        statistic=agg_results.iloc[index, 0],
                                        num_items=agg_results['num_items'].iloc[index]))
    return sample_results


  async def execute_inference_and_get_results_on_sampled_data(
      self,
      sampler: UniformBlobSampler,
      all_blobs_df: pd.DataFrame,
      num_samples: int,
      query: Query,
      conn
  ):
    sample_blobs = sampler.sample_next_n(num_samples=num_samples)

    sample_index = [int(sample.blob_id.iloc[0]) for sample in sample_blobs]
    sample_blob_key_df = all_blobs_df.iloc[sample_index]

    await self.execute_inference_services(query, sample_blob_key_df, conn)

    agg_results = await self.get_agg_results(query, sample_blob_key_df, conn)

    sample_results = self.get_sample_results(sample_blob_key_df, sample_blobs, agg_results)
    return sample_results


  def get_additional_required_num_samples(
      self,
      query: Query,
      sample_results: List[SampledBlob],
      estimator:Estimator
  ) -> int:
    error_target = query.error_target
    conf = query.confidence / 100.
    alpha = 1. - conf
    pilot_estimate = estimator.estimate(sample_results, _NUM_PILOT_SAMPLES, conf / 2)
    p_lb = statsmodels.stats.proportion.proportion_confint(len(sample_results), _NUM_PILOT_SAMPLES, alpha / 2.)[0]
    num_samples = int(
      (scipy.stats.norm.ppf(alpha / 2) * pilot_estimate.std_ub / error_target) ** 2 * \
      (1. / p_lb)
    )

    return num_samples
