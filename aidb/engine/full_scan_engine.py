import pandas as pd
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''

    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order

    inference_services_executed = set()
    for bound_service in service_ordering:
      inp_query_str = self.get_input_query_for_inference_service(bound_service, query, inference_services_executed)
      print(f'inp_query_str: {inp_query_str}')

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))
        print(f'inp_df: {inp_df}')
      # The bound inference service is responsible for populating the database
      bs_res = await bound_service.infer(inp_df)
      print(f'bs_res: {bs_res}')

      inference_services_executed.add(bound_service.service.name)

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql_query_text))
      return res.fetchall()
