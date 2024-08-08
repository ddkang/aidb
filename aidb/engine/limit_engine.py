import numpy as np
import pandas as pd
from sqlalchemy.sql import text
from typing import List

from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query
from aidb.utils.constants import VECTOR_ID_COLUMN

class LimitEngine(TastiEngine):
  async def _execute_limit_query(self, query: Query):
    '''
    execute service inference based on proxy score, stop when the limit number meets
    '''
    # generate proxy score for each blob
    proxy_score_for_all_blobs = await self.get_proxy_scores_for_all_blobs(query)

    # sorted blob id based on proxy score
    id_score = [(i, s) for i, s in zip(proxy_score_for_all_blobs.index, proxy_score_for_all_blobs.values)]
    sorted_list = sorted(id_score, key=lambda x: x[1], reverse=True)
    desired_cardinality = query.limit_cardinality

    # TODO: rewrite query, use full scan to execute query
    bound_service_list = query.inference_engines_required_for_query

    limit_engine_batch_size = len(sorted_list)
    for bound_service in bound_service_list:
      limit_engine_batch_size = min(bound_service.service.preferred_batch_size, limit_engine_batch_size)

    batched_indexes_list = [
        [item[0] for item in sorted_list[i:i + limit_engine_batch_size]]
        for i in range(0, len(sorted_list), limit_engine_batch_size)
    ]

    for batched_indexes in batched_indexes_list:
      for bound_service in bound_service_list:
        inp_query_str = self.get_input_query_for_inference_service_filtered_index(bound_service,
                                                                                  self.blob_mapping_table_name,
                                                                                  batched_indexes)
        async with self._sql_engine.begin() as conn:
          inp_df = await conn.run_sync(lambda conn: pd.read_sql_query(text(inp_query_str), conn))
        inp_df.set_index(VECTOR_ID_COLUMN, inplace=True, drop=True)
        await bound_service.infer(inp_df)

      # FIXME: Currently, we select the whole database, need to rewrite sql text to select specific blob id
      async with self._sql_engine.begin() as conn:
        res = await conn.execute(text(query.sql_query_text))
      res = res.fetchall()

      if len(res) == desired_cardinality:
        break

    return res
