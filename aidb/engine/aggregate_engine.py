import pandas as pd
from sqlalchemy.sql import text
import sqlglot.expressions as exp
from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query
from aidb.sampler.sampler import Sampler


class AggregateEngine(BaseEngine):

  async def execute_aggregate_query(self, query):
    query.process_aggregate_query()    

    service_ordering = self._config.inference_topological_order
    for bound_service in service_ordering:
      binding = bound_service.binding
      self.inp_cols = binding.input_columns
      self.inp_cols_str = ', '.join(self.inp_cols)
      self.inp_tables = self.get_tables(self.inp_cols)
      self.join_str = self.get_join_str(self.inp_tables)
      self.sampler = Sampler()

      query.join_str = self.get_join_str(query.tables_concerned)
      num_ids_query = self.sampler.get_num_blob_ids_query(
                              query.aggregated_column,
                              ','.join(query.columns_concerned),
                              query.tables_concerned,
                              query.join_str
                      )
      async with self._sql_engine.begin() as conn:
        self.num_blob_ids = await conn.run_sync(
                lambda conn: pd.read_sql(text(num_ids_query), conn)
              )
        self.num_blob_ids = self.num_blob_ids.iloc[0, 0]
        inp_query_str, scaling_factor = self.sampler.get_random_sampling_query(
                                                      self.inp_cols_str,
                                                      self.inp_tables,
                                                      self.join_str,
                                                      self.num_blob_ids
                                              )

        inp_df = await conn.run_sync(
                lambda conn: pd.read_sql(text(inp_query_str), conn)
              )
      await bound_service.infer(inp_df)

    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql))
      res = res.fetchall()[0][0]
      if query.agg_type == exp.Count or query.agg_type == exp.Sum:
        res *= scaling_factor
      return res