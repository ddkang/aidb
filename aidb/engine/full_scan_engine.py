import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: str, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    for bound_service in service_ordering:
      binding = bound_service.binding
      inp_cols = binding.input_columns
      inp_cols_str = ', '.join(inp_cols)
      inp_tables = self.get_tables(inp_cols)
      join_str = self.get_join_str(inp_tables)
      inp_query_str = f'''
        SELECT {inp_cols_str}
        FROM {', '.join(inp_tables)}
        {join_str};
      '''
      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query))
      return res.fetchall()