import pandas as pd
from sqlalchemy.sql import text
from typing import List

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query

QUERY_LIMIT = 10000


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
    is_udf_query = query.is_udf_query
    if is_udf_query:
      dataframe_sql, query = query.udf_query_extraction

    bound_service_list = query.inference_engines_required_for_query
    inference_services_executed = set()
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filter_service(bound_service,
                                                                                query,
                                                                                inference_services_executed)

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

      inference_services_executed.add(bound_service.service.name)

    async with self._sql_engine.begin() as conn:
      if is_udf_query:
        offset = 0
        res = []
        while True:
          query = query.add_limit_keyword(QUERY_LIMIT)
          query = query.add_offset_keyword(offset)
          res_df = await conn.run_sync(lambda conn: pd.read_sql(text(query.sql_query_text), conn))
          res_satisfy_udf = self.execute_user_defined_function(res_df, dataframe_sql, query)
          res.extend(res_satisfy_udf)
          if len(res_df) != QUERY_LIMIT:
            return res
          offset += QUERY_LIMIT

      else:
        res = await conn.execute(text(query.sql_query_text))
        return res.fetchall()