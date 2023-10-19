import pandas as pd
from sqlalchemy.sql import text
from typing import List

from aidb.engine.base_engine import BaseEngine
from aidb.inference.bound_inference_service import BoundInferenceService
from aidb.query.query import Query

class FullScanEngine(BaseEngine):
  def _get_required_bound_services_order(self, query: Query) -> List[BoundInferenceService]:
    bound_service_list_ordered = [
        bound_service
        for bound_service in self._config.inference_topological_order
        if bound_service in query.inference_engines_required_for_query
    ]
    return bound_service_list_ordered


  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
    bound_service_list = self._get_required_bound_services_order(query)
    inference_services_executed = set()
    for bound_service in bound_service_list:
      inp_query_str, _, _, _ = self.get_input_query_for_inference_service_filter_service(bound_service,
                                                                                         query,
                                                                                         inference_services_executed)

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

      inference_services_executed.add(bound_service.service.name)

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql_query_text))
      return res.fetchall()
