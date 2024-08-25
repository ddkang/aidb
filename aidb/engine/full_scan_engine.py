import pandas as pd
from sqlalchemy.sql import text
from typing import List
from collections import defaultdict

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query
from aidb.utils.order_optimization_utils import (
  get_currently_supported_filtering_predicates_for_ordering, 
  reorder_inference_engine
)

RETRIEVAL_BATCH_SIZE = 10000


          
class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
    
    bound_service_list = query.inference_engines_required_for_query
    supported_filtering_predicates = get_currently_supported_filtering_predicates_for_ordering(self._config, query)
    engine_to_proxy_score = {}
    bound_service_list = query.inference_engines_required_for_query
    for engine, related_predicates in supported_filtering_predicates.items():
      engine_fp = self._get_where_str(related_predicates)
      adjusted_query = f'select * from {engine} WHERE {engine_fp}'
      adjusted_query = Query(adjusted_query, self._config)
      # proxy_score_for_all_blobs = await self.get_proxy_scores_for_all_blobs(adjusted_query, return_binary_score=True)
      # engine_to_proxy_score[engine] = proxy_score_for_all_blobs.sum() / len(proxy_score_for_all_blobs)
      engine_to_proxy_score[engine] = 1
    bound_service_list = reorder_inference_engine(engine_to_proxy_score, bound_service_list)
    is_udf_query = query.is_udf_query
    if is_udf_query:
      query.check_udf_query_validity()
      dataframe_sql, query = query.udf_query

    inference_services_executed = set()
    for bound_service in bound_service_list:
      print(70, bound_service)
      inp_query_str = self.get_input_query_for_inference_service_filter_service(bound_service,
                                                                                query,
                                                                                inference_services_executed)
      print(73, inp_query_str)
      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(inp_query_str), conn))
      print(76, inp_df)
      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

      inference_services_executed.add(bound_service.service.name)

    async with self._sql_engine.begin() as conn:
      if is_udf_query:
        offset = 0
        res = []
        # selectively load data from database to avoid large memory usage
        while True:
          query = query.add_limit_keyword(RETRIEVAL_BATCH_SIZE)
          query = query.add_offset_keyword(offset)
          res_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(query.sql_query_text), conn))
          row_length = len(res_df)
          res_satisfy_udf = self.execute_user_defined_function(res_df, dataframe_sql, query)
          res.extend([tuple(row) for row in res_satisfy_udf.itertuples(index=False)])
          if row_length != RETRIEVAL_BATCH_SIZE:
            return res
          offset += RETRIEVAL_BATCH_SIZE

      else:
        res = await conn.execute(text(query.sql_query_text))
        return res.fetchall()