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


def get_flat_keys(cols):
  return [col.split('.')[1] for col in cols]

def combine_str(cols):
  return ','.join(cols)


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
  
  def get_pilot_sampling_query(self,bound_service, query,
                        inference_services_executed):
    inp_query_str = self.get_input_query_for_inference_service(
                          bound_service,
                          query,
                          inference_services_executed
                        )

    return f'''{inp_query_str.replace(';', '')}
                ORDER BY RANDOM()
                LIMIT {self.num_pilot_samples};'''


  def get_num_blob_ids_query(self, table, column, num_samples=100):
    num_ids_query = f'''
                      SELECT COUNT({column})
                      FROM {table};
                      '''
    return num_ids_query

  def get_population_percent(self):
    return 10

  def statistic_fn(self, sample, agg_on_column, agg_table):
    column_of_interest = f'{agg_table}.{agg_on_column}'
    print(f'sample: {sample}, typesample: {type(sample)}')
    print(f'column_of_interest: {column_of_interest}')
    if column_of_interest not in sample.columns:
      print('cond1')
      return False
    if sample[column_of_interest].empty:
      print('cond2')
      return False
    return True

  async def get_sampled_blobs_with_stats(self, infer_output, agg_on_column, agg_table, query, another_conn):
    sampled_blobs = []
    num_samples = len(infer_output)
    mass = 1.
    wt = 1. / (num_samples)
    for idx in range(num_samples):
      print(f'idx: {idx}')
      print(f'{repr(infer_output[idx])}, type: {type(infer_output[idx])}')
      sampled_blob_id = SampledBlobId(
                          infer_output[idx],
                          1. / num_samples,
                          1.
                        )
      blob_table = {agg_table: infer_output[idx]} # self._config.blob_tables[0]
      statistic_ans = self.statistic_fn(infer_output[idx], agg_on_column, agg_table) # if statistic_fn is not None else None
      if not statistic_ans:
        continue

      map_cols = {}
      for col in infer_output[idx].columns:
        map_cols[col] = col.split('.')[1]
      infer_output[idx].rename(columns=map_cols, inplace=True)
      print(f'modified df: {repr(infer_output[idx])}, type: {type(infer_output[idx])}')
      proxy_table = f'{agg_table}_proxy'

      await another_conn.run_sync(
            lambda sync_conn: infer_output[idx].to_sql(
                proxy_table,
                con=sync_conn,
                if_exists='replace',
                index=False
            )
        )
      new_query = query.sql_query_text.replace(agg_table, proxy_table)
      print(f'orig q:{query.sql_query_text}, query executed: {new_query}')
      res = await another_conn.execute(text(new_query))
      print(f'res: {res},')
      res = res.fetchone()
      print(f' res.fetchone/ statistic_ans: {res}')      
      statistic_ans = res[0]

      sampled_blobs.append(SampledBlob(
                              infer_output[idx],
                              1. / num_samples,
                              1.,
                              blob_table,
                              statistic_ans,
                              {agg_table: len(infer_output[idx])} # self._config.blob_tables[0]
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
      print(f'''inp_col: {inp_col}
                inp_table: {inp_table}
                    ''')
      print('num_ids_query', num_ids_query)
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
    print('self.num_blob_ids: ', self.num_blob_ids, type(self.num_blob_ids))
    self.num_pilot_samples = int(self.num_blob_ids * population_percent / 100.)

    service_ordering = self._config.inference_topological_order
    print(f'service_ordering: {service_ordering}')
    inference_services_executed = set()
    print(f'self._config.column_by_service: {self._config.column_by_service}')
    async with self._sql_engine.begin() as conn:
      for bound_service in service_ordering:
        out_query = self.get_pilot_sampling_query(
                            bound_service,
                            query,
                            inference_services_executed
                          )
        print(f'out_query: {out_query}')
        out_df = await conn.run_sync(
                          lambda conn: pd.read_sql(
                            text(out_query),
                            conn
                          )
                        )
        print('out_df: ', out_df.head(), len(out_df))

        if not f'{agg_on_column_table}.{agg_on_column}' in bound_service.binding.output_columns:
          continue
        print(f'agg_on_column_table: {agg_on_column_table}, agg_on_column: {agg_on_column}')
        input_samples = await bound_service.infer(out_df)
        print(f'input_samples: {repr(input_samples)}, type: {type(input_samples)}')

        sampled_blobs = await self.get_sampled_blobs_with_stats(input_samples, agg_on_column, agg_on_column_table, query, conn)
        print(f'sampled_blobs: {repr(sampled_blobs)}')

        estimator = self._get_estimator(agg_type, columns)
        pilot_estimate = estimator.estimate(sampled_blobs, self.num_pilot_samples, conf / 2, agg_table=agg_on_column_table)
        print(f'pilot_estimate: {repr(pilot_estimate)}')
        p_lb = statsmodels.stats.proportion.proportion_confint(len(sampled_blobs), self.num_pilot_samples, alpha / 2.)[0]
        num_samples = int(
          (scipy.stats.norm.ppf(alpha / 2) * pilot_estimate.std_ub / error_target) ** 2 * \
            (1. / p_lb)
        )
        print(f'pilot_estimate.std_ub, num_samples, p_lb: {pilot_estimate}, {pilot_estimate.std_ub}, {num_samples}, {p_lb}')

        self.num_pilot_samples = num_samples # change variable name, as now we're using final no. of samples
        final_query = self.get_pilot_sampling_query(
                            bound_service,
                            query,
                            inference_services_executed
                          )
        print(f'final_query: {final_query}')
        final_df = await conn.run_sync(
                          lambda conn: pd.read_sql(
                            text(final_query),
                            conn
                          )
                        )

        print('final_df: ', final_df.head(), len(final_df))

        #ERROR occuring at below line of code
        '''sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked
          [SQL: INSERT INTO objects00 (frame, object_name, confidence, x_min, y_min, x_max, y_max, object_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT (frame, object_id) DO UPDATE SET frame = ?, object_name = ?, confidence = ?, x_min = ?, y_min = ?, x_max = ?, y_max = ?, object_id = ?]
          [parameters: (2040, 'car', 1.0, 1144.31, 821.11, 1590.52, 1077.12, 0, 2040, 'car', 1.0, 1144.31, 821.11, 1590.52, 1077.12, 0)]
          (Background on this error at: https://sqlalche.me/e/14/e3q8)
        '''
        all_samples = await bound_service.infer(final_df)
        input_samples.extend(all_samples)
        all_blobs = await self.get_sampled_blobs_with_stats(input_samples, agg_on_column, agg_on_column_table, query, conn)
        print(f'all_blobs needed: {repr(all_blobs)}, len(all_blobs): {len(all_blobs)}')
        return [(estimator.estimate(all_blobs, num_samples, conf, agg_table=agg_on_column_table).estimate,)]