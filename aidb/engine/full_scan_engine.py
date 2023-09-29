from typing import List

import pandas as pd
from sqlalchemy.dialects.mysql import insert
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query

import networkx as nx


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

    def get_original_column(column, column_graph: nx.DiGraph):
      if column not in column_graph.nodes:
        return column
      column_derived_from = list(column_graph[column].keys())
      assert len(column_derived_from) <= 1
      if len(column_derived_from) == 0:
        return column
      else:
        return get_original_column(column_derived_from[0], column_graph)


    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    filtering_predicates = query.filtering_predicates
    columns_by_service = self._config.column_by_service
    column_graph = self._config.column_graph
    inference_engines_required_predicates = []
    for filtering_predicate in filtering_predicates:
      inference_engines_required = set()
      for or_connected_predicate in filtering_predicate:
        if or_connected_predicate.left_exp.type == "column":
          originated_from = get_original_column(or_connected_predicate.left_exp.value, column_graph)
          if originated_from in columns_by_service:
            inference_engines_required.add(columns_by_service[originated_from].service.name)
      inference_engines_required_predicates.append(inference_engines_required)


    for bound_service in service_ordering:
      binding = bound_service.binding
      inp_cols = binding.input_columns
      refined_inp_cols = [get_original_column(col, column_graph) for col in inp_cols]
      inp_cols_str = ', '.join(refined_inp_cols)
      inp_tables = get_tables(refined_inp_cols)
      join_str = get_join_str(inp_tables)
      inp_query_str = f'''
        SELECT {inp_cols_str}
        FROM {', '.join(inp_tables)}
        {join_str};
      '''

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      inp_df.columns = inp_cols

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.get_sql_query_text()))
      return res.fetchall()