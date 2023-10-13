from sqlalchemy.sql import text
import pandas as pd

from aidb.engine.tasti_engine import TastiEngine
from aidb.query.query import Query


class LimitEninge(TastiEngine):
  async def _execute_limit_query(self, query: Query):
    '''
    execute service inference based on proxy score, stop when the limit number meets
    '''
    reps_df, proxy_score_for_all_blobs = await self.execute_tasti(query)
    executed_index = set(reps_df.index)
    id_score = [(i, s) for i, s in zip(proxy_score_for_all_blobs.index, proxy_score_for_all_blobs.values)]
    sorted_list = sorted(id_score, key=lambda x: x[1], reverse=True)
    desired_cardinality = int(query.get_limit_cardinality())

    bound_service_list = self._get_required_bound_services_order(query)
    for index, _ in sorted_list:
      if index in executed_index:
        continue
      executed_index.add(index)

      for bound_service in bound_service_list:
        inp_query_str = self.get_input_query_for_inference_service(bound_service, query,
                                                                   self.blob_mapping_table_name, [index])
        async with self._sql_engine.begin() as conn:
          inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
        inp_df.set_index('blob_id', inplace=True, drop=True)
        await bound_service.infer(inp_df)

      # FIXME: Currently, we select the whole database, need to rewrite sql text to select specific blob id
      async with self._sql_engine.begin() as conn:
        res = await conn.execute(text(query.sql_query_text))
      res = res.fetchall()

      if len(res) == desired_cardinality:
        break

    return res
