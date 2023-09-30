from typing import List

import networkx as nx
import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine


class FullScanEngine(BaseEngine):
  async def execute_full_scan(self, query: str, **kwargs):
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
      table_graph = self._config.table_graph.subgraph(tables)
      # all input tables should be connectable
      # NetworkX is connected not implemented for directed graph
      assert nx.is_connected(table_graph.to_undirected())
      table_order = list(nx.topological_sort(table_graph))[::-1]
      table_pairs = []
      for table in table_order:
         table_pairs += [(e[1], e[0]) for e in table_graph.in_edges(table)]
      first_table = table_order[0]
      # table_pairs = [(tables[i], tables[i+1]) for i in range(len(table_order) - 1)]
      join_strs = []
      for table1_name, table2_name in table_pairs:
        table2 = self._config.tables[table2_name]
        join_cols = []
        for _, fkey in table2.foreign_keys.items():
          if fkey.startswith(table1_name + '.'):
            join_cols.append((fkey, fkey.replace(table1_name, table2_name, 1)))
        if len(join_cols) > 0:
          join_strs.append(f'INNER JOIN {table2_name} ON {" AND ".join([f"{col1} = {col2}" for col1, col2 in join_cols])}')
      return f"FROM {first_table}\n" + '\n'.join(join_strs)


    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    for bound_service in service_ordering:
      binding = bound_service.binding
      inp_cols = binding.input_columns
      inp_cols_str = ', '.join(inp_cols)
      inp_tables = get_tables(inp_cols)
      join_str = get_join_str(inp_tables)
      inp_query_str = f'''
        SELECT {inp_cols_str}
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