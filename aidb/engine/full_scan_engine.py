from collections import defaultdict
from typing import List

import pandas as pd
from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query
from aidb.utils.logger import logger
from aidb.utils.order_optimization_utils import (
    get_currently_supported_filtering_predicates_for_ordering,
    reorder_inference_engine)
from sqlalchemy.sql import text

RETRIEVAL_BATCH_SIZE = 10000


          
class FullScanEngine(TastiEngine):
  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
  
    is_udf_query = query.is_udf_query
    if is_udf_query:
      query.check_udf_query_validity()
      dataframe_sql, query = query.udf_query

    bound_service_list = query.inference_engines_required_for_query
    if self.tasti_index:
      supported_filtering_predicates = get_currently_supported_filtering_predicates_for_ordering(self._config, query)
      engine_to_proxy_score = {}
      bound_service_list = query.inference_engines_required_for_query
      for engine, related_predicates in supported_filtering_predicates.items():
        where_str = self._get_where_str(related_predicates)
        select_join_str = f'select * from {engine.service.name} '
        if len(related_predicates) > 0:
          adjusted_query = select_join_str + f'WHERE {where_str};'
        else:
          adjusted_query = select_join_str + ';'

        adjusted_query = Query(adjusted_query, self._config)
        proxy_score_for_all_blobs = await self.get_proxy_scores_for_all_blobs(adjusted_query, return_binary_score=True)
        engine_to_proxy_score[engine] = proxy_score_for_all_blobs.sum() / len(proxy_score_for_all_blobs)
      logger.info(f'Proxy scores for all engines: {engine_to_proxy_score}')
      bound_service_list = reorder_inference_engine(engine_to_proxy_score, bound_service_list)
      logger.info(f'The order of inference engines: {bound_service_list}')
      
    inference_services_executed = set()
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service_filter_service(bound_service,
                                                                                query,
                                                                                inference_services_executed)

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(inp_query_str), conn))

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