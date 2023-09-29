from typing import List

import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: Query, **kwargs):
    '''
    Executes a query by doing a full scan and returns the results.
    '''
    def get_tables(columns: List[str]) -> List[str]:
      tables = set()
      for col in columns:
        table_name = col.split('.')[0]
        tables.add(table_name)
      return list(tables)

    def get_join_str(tables: List[str]) -> str:
      table_pairs = [(tables[i], tables[i+1]) for i in range(len(tables) - 1)]
      join_strs = []
      for table1_name, table2_name in table_pairs:
        table1 = self._config.tables[table1_name]
        join_cols = []
        for fkey in table1.foreign_keys:
          if fkey.startswith(table2_name + '.'):
            join_cols.append((fkey, fkey.replace(table2_name, table1_name, 1)))
        join_strs.append(f'INNER JOIN {table2_name} ON {" AND ".join([f"{col1} = {col2}" for col1, col2 in join_cols])}')
      return '\n'.join(join_strs)


    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    filtering_predicates = query.filtering_predicates
    for bound_service in service_ordering:
      binding = bound_service.binding
      inp_cols = binding.input_columns
      inp_cols_str = ', '.join(inp_cols)
      inp_tables = get_tables(inp_cols)
      join_str = get_join_str(inp_tables)
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
      res = await conn.execute(text(query.get_sql_query_text()))
      return res.fetchall()