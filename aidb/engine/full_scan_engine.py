import pandas as pd
from typing import Optional, List

from sqlalchemy.sql import text
from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query
from aidb.inference.bound_inference_service import BoundInferenceService

class FullScanEngine(BaseEngine):
  async def execute_query(self, query_str: str, set_index: bool = False, index_column: Optional[str] = None):
    async with self._sql_engine.begin() as conn:
      df = await conn.run_sync(lambda conn: pd.read_sql(text(query_str), conn))
    if set_index:
      df.set_index(index_column, inplace=True, drop=True)
    return df


  def _get_required_bound_services_order(self, query: Query) -> List[BoundInferenceService]:
    service_ordering = self._config.inference_topological_order
    bound_service_list = query.inference_engines_required_for_query
    bound_service_list_ordered = []
    for bound_service in service_ordering:
      if bound_service in bound_service_list:
        bound_service_list_ordered.append(bound_service)
    return bound_service_list_ordered


  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
    bound_service_list = self._get_required_bound_services_order(query)
    for bound_service in bound_service_list:
      inp_query_str = self.get_input_query_for_inference_service(bound_service, query)
      inp_df = await self.execute_query(inp_query_str)

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)


    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql_query_text))
      return res.fetchall()
