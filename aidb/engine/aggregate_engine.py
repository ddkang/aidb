from collections import defaultdict

import pandas as pd
import scipy
import sqlglot.expressions as exp
import statsmodels.stats.proportion
from aidb.engine.base_engine import BaseEngine
from aidb.estimator.estimator import (
  Estimator, WeightedCountSetEstimator, WeightedMeanSetEstimator, WeightedSumSetEstimator)
from aidb.query.query import Query
from aidb.samplers.sampler import SampledBlob
from aidb.utils.constants import (
  ESTIMATE_AGG_RESULTS_MODE, FIND_NUM_SAMPLES_MODE, NUM_PILOT_SAMPLES, NUM_SAMPLES_SPLIT,
  ESTIMATE_NUM_SAMPLES_NORMAL_MODE, ESTIMATE_NUM_SAMPLES_WILSON_MODE)
from sqlalchemy.sql import text
from sqlglot.generator import Generator


def csv(*args, sep=", "):
  return sep.join(arg for arg in args if arg)


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


  def query_select_all_exp(self, query, chunk_size, offset):
    gen = Generator()
    query_exp = query.get_expression()
    query_limit = query.limit_cardinality
    query_offset = query.offset_number
    limit = chunk_size
    if query_limit:
      if offset + chunk_size <= query_limit:
        limit = chunk_size
      else:
        limit = query_limit - offset
    if query_offset:
      offset += query_offset
    select_all_str = csv(
      f"SELECT *",
      gen.sql(query_exp, "from"),
      *[gen.sql(sql) for sql in query_exp.args.get("laterals", [])],
      *[gen.sql(sql) for sql in query_exp.args.get("joins", [])],
      gen.sql(query_exp, "where"),
      gen.sql(query_exp, "group"),
      gen.sql(query_exp, "having"),
      gen.sql(query_exp, "order"),
      f" LIMIT {limit} ",
      f"OFFSET {offset}",
      sep="",
    )
    return select_all_str


  def get_select_exp_str(self, query):
    gen = Generator()
    query_exp = query.get_expression()
    hint = gen.sql(query_exp, "hint")
    distinct = " DISTINCT" if query_exp.args.get("distinct") else ""
    expressions = gen.expressions(query_exp)
    select = "SELECT" if expressions else ""
    sep = gen.sep() if expressions else ""
    return f"{select}{hint}{distinct}{sep}{expressions}"


  def get_random_sampling_query(self, bound_service, query, inference_services_executed, num_samples):
    """Function to return limited number of samples randomly"""
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
    """Function that returns a query, to get total number of blob ids"""
    return f'''
            SELECT COUNT(*)
            FROM {table};
          '''


  def get_aggregation_query_for_chunk(self, query, limit, offset, select_exp_str):
    '''
    Returns aggregation queries on few chunks of data as per limit, offset
    Eg: select avg(x_min)
        from
        (select x_min from objects00 limit 10 offset 20);
    '''
    select_all_str = self.query_select_all_exp(query, limit, offset)
    return f'{select_exp_str} FROM({select_all_str})'


  def find_num_required_samples(self, estimator, agg_stats, conf, alpha, error_target, num_column_samples):
    z_score = scipy.stats.norm.ppf(alpha / 2.)
    method = ESTIMATE_NUM_SAMPLES_NORMAL_MODE if num_column_samples / NUM_PILOT_SAMPLES > 0.5 else ESTIMATE_NUM_SAMPLES_WILSON_MODE
    population_lb = statsmodels.stats.proportion.proportion_confint(
      num_column_samples, NUM_PILOT_SAMPLES, alpha,
      method
      )[0]
    pilot_estimate = estimator.estimate(
      agg_stats,
      conf,
      num_column_samples=num_column_samples
    )
    num_samples = ((z_score ** 2) * (pilot_estimate.std_ub ** 2)) / (error_target ** 2)
    num_samples = num_samples / population_lb
    return int(num_samples)


  async def populate_samples_of_concerned_services(self, inference_services_required, query, conn, num_samples):
    """
    Function to infer data corresponding to a subset of ids that satisfy query conditions
    and group results service wise
    """
    num_ids_per_service = defaultdict(lambda: 0)
    inference_services_executed = set()
    for bound_service in self._config.inference_topological_order:
      if bound_service not in inference_services_required:
        continue
      table_sampling_query = self.get_random_sampling_query(
        bound_service,
        query,
        inference_services_executed,
        num_samples
      )
      async with self._sql_engine.begin() as conn:
        sampled_df = await conn.run_sync(lambda conn: pd.read_sql(text(table_sampling_query), conn))
      samples_from_service = await bound_service.infer(sampled_df)
      num_ids_per_service[bound_service.service.name] = len(samples_from_service)
      inference_services_executed.add(bound_service.service.name)
    return num_ids_per_service


  async def execute_query_on_data_chunks(self, infer_ouput_length, est_conn, query):
    """
    Returns list of aggregation results on chunks of data
    2D list with row - chunk id, col - agg type
    """
    aggregation_data = []
    chunk_size = max(infer_ouput_length // NUM_SAMPLES_SPLIT, 1)
    offset = 0
    num_chunks_processed = 0
    while num_chunks_processed < NUM_SAMPLES_SPLIT:
      select_exp_str = self.get_select_exp_str(query)
      new_query = self.get_aggregation_query_for_chunk(query, chunk_size, offset, select_exp_str)
      chunk_statistics = await est_conn.execute(text(new_query))
      chunk_statistics = chunk_statistics.fetchall()

      # since there are no nested queries, take index 0
      aggregation_data.append(chunk_statistics[0])
      offset += chunk_size
      num_chunks_processed += 1
    return aggregation_data, chunk_size


  def get_agg_results_on_chunks(self, query, num_ids_per_service, sampled_chunks, chunk_size):
    """
    Return aggregation results with weights, which can be approximated upon
    """
    results = defaultdict(lambda: defaultdict(lambda: None))
    agg_col_index = 0
    for agg_type, columns_per_agg in query.aggregated_columns_and_types:
      for columns in columns_per_agg:
        agg_stats = []
        num_column_samples = num_ids_per_service[query.config.column_by_service[columns[0]].service.name]
        mass = 1.
        wt = NUM_SAMPLES_SPLIT / (num_column_samples / chunk_size)
        for chunk_statistics in sampled_chunks:
          statistic_ans = chunk_statistics[agg_col_index]
          if statistic_ans is None: continue
          agg_stats.append(SampledBlob(wt, mass, statistic_ans, chunk_size))
        results[agg_type].update(
          {
            columns[0]: agg_stats}
          )
        agg_col_index += 1
    return results


  def get_estimates_on_data_chunks_agg_column_wise(
      self,
      query,
      agg_results_on_chunks,
      mode,
      num_ids_per_service=None,
      num_samples=None
  ):
    results = []
    error_target = query.error_target
    conf = query.confidence / 100.
    alpha = 1. - conf
    for agg_type, columns_per_agg in query.aggregated_columns_and_types:
      estimator = self._get_estimator(agg_type)
      for columns in columns_per_agg:
        agg_column_data = agg_results_on_chunks[agg_type][columns[0]]
        if mode == FIND_NUM_SAMPLES_MODE:
          num_column_samples = num_ids_per_service[query.config.column_by_service[columns[0]].service.name]
          results.append(
            self.find_num_required_samples(estimator, agg_column_data, conf, alpha, error_target, num_column_samples)
          )
        elif mode == ESTIMATE_AGG_RESULTS_MODE:
          results.append(estimator.estimate(agg_column_data, conf, num_column_samples=num_samples).estimate)
    return results


  async def execute_aggregate_query(self, query: Query):
    """
    Execute aggregation query using approximate processing
    """
    inference_services_required = set([query.config.column_by_service[column] for column in query.columns_in_query])
    self.num_blob_ids = {}
    for blob_table in self._config.blob_tables:
      async with self._sql_engine.begin() as conn:
        num_ids = await conn.execute(text(self.get_num_blob_ids_query(blob_table)))
      self.num_blob_ids[blob_table] = num_ids.fetchone()[0]

    # Pilot run and inference to determine required number of samples
    async with self._sql_engine.begin() as conn:
      num_ids_per_service = await self.populate_samples_of_concerned_services(
        inference_services_required,
        query,
        conn,
        NUM_PILOT_SAMPLES
      )
      total_num_ids = sum(num_ids_per_service.values())
      sampled_chunks, chunk_size = await self.execute_query_on_data_chunks(total_num_ids, conn, query)

    agg_results_on_chunks = self.get_agg_results_on_chunks(query, num_ids_per_service, sampled_chunks, chunk_size)
    num_samples_per_agg_column = self.get_estimates_on_data_chunks_agg_column_wise(
      query,
      agg_results_on_chunks,
      mode=FIND_NUM_SAMPLES_MODE,
      num_ids_per_service=num_ids_per_service
    )
    num_samples = max(NUM_SAMPLES_SPLIT, max(num_samples_per_agg_column))

    # Final query execution on required number of samples
    async with self._sql_engine.begin() as conn:
      num_ids_sampled_per_service = await self.populate_samples_of_concerned_services(
        inference_services_required,
        query,
        conn,
        num_samples
      )
      total_num_ids_sampled = sum(num_ids_sampled_per_service.values())
      sampled_chunks, chunk_size = await self.execute_query_on_data_chunks(total_num_ids_sampled, conn, query)

    agg_results_on_chunks = self.get_agg_results_on_chunks(
      query, num_ids_sampled_per_service, sampled_chunks, chunk_size
      )
    all_samples_estimates = self.get_estimates_on_data_chunks_agg_column_wise(
      query,
      agg_results_on_chunks,
      mode=ESTIMATE_AGG_RESULTS_MODE,
      num_samples=len(sampled_chunks) * chunk_size
    )
    return [tuple(all_samples_estimates)]