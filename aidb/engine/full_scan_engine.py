from typing import List

import pandas as pd
import sqlglot.expressions as exp
from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query, FilteringClause

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
      table_graph = self._config.table_graph.subgraph(tables)
      # all input tables should be connectable
      # NetworkX is connected not implemented for directed graph
      assert nx.is_connected(table_graph.to_undirected())
      table_order = list(nx.topological_sort(table_graph))[::-1]
      table_pairs = []
      for table in table_order:
        table_pairs += [(e[1], e[0]) for e in table_graph.in_edges(table)]
      first_table = table_order[0]
      join_strs = []
      for table1_name, table2_name in table_pairs:
        table2 = self._config.tables[table2_name]
        join_cols = []
        for _, fkey in table2.foreign_keys.items():
          if fkey.startswith(table1_name + '.'):
            join_cols.append((fkey, fkey.replace(table1_name, table2_name, 1)))
        if len(join_cols) > 0:
          join_strs.append(
            f'INNER JOIN {table2_name} ON {" AND ".join([f"{col1} = {col2}" for col1, col2 in join_cols])}')
      return f"FROM {first_table}\n" + '\n'.join(join_strs)

    def get_original_column(column, column_graph: nx.DiGraph):
      if column not in column_graph.nodes:
        return column
      column_derived_from = list(column_graph[column].keys())
      assert len(column_derived_from) <= 1
      if len(column_derived_from) == 0:
        return column
      else:
        return get_original_column(column_derived_from[0], column_graph)

    def get_expr_string(node):
      if node == exp.GT:
        return ">"
      elif node == exp.LT:
        return "<"
      elif node == exp.GTE:
        return ">="
      elif node == exp.LTE:
        return "<="
      elif node == exp.EQ:
        return "="
      elif node == exp.Like:
        return "LIKE"
      elif node == exp.NEQ:
        return "!="
      else:
        raise NotImplementedError

    def predicate_to_str(p: FilteringClause):
      if p.right_exp is None:
        sql_expr = str(p.left_exp.value)
      else:
        sql_expr = f"{p.left_exp.value} {get_expr_string(p.op)} {p.right_exp.value}"
      if p.is_negation:
        sql_expr = "NOT " + sql_expr
      return sql_expr

    def get_where_str(filtering_predicates: List[List[FilteringClause]]):
      and_connected = []
      for fp in filtering_predicates:
        and_connected.append(" OR ".join([predicate_to_str(p) for p in fp]))
      return " AND ".join(and_connected)

    def get_inference_engines_required(filtering_predicates, columns_by_service):
      inference_engines_required_predicates = []
      for filtering_predicate in filtering_predicates:
        inference_engines_required = set()
        for or_connected_predicate in filtering_predicate:
          if or_connected_predicate.left_exp.type == "column":
            originated_from = get_original_column(or_connected_predicate.left_exp.value, column_graph)
            if originated_from in columns_by_service:
              inference_engines_required.add(columns_by_service[originated_from].service.name)
        inference_engines_required_predicates.append(inference_engines_required)
      return inference_engines_required_predicates

    # The query is irrelevant since we do a full scan anyway
    service_ordering = self._config.inference_topological_order
    filtering_predicates = query.filtering_predicates
    columns_by_service = self._config.column_by_service
    column_graph = self._config.column_graph
    inference_engines_required_predicates = get_inference_engines_required(filtering_predicates, columns_by_service)

    inference_services_executed = set()
    for bound_service in service_ordering:
      binding = bound_service.binding
      inp_cols = binding.input_columns
      original_column_mapping = {}
      refined_inp_cols = []
      for col in inp_cols:
        original_column = get_original_column(col, column_graph)
        refined_inp_cols.append(original_column)
        original_column_mapping[col] = original_column

      inp_cols_str = ', '.join(refined_inp_cols)
      inp_tables = get_tables(refined_inp_cols)
      join_str = get_join_str(inp_tables)

      filtering_predicates_satisfied = []
      for p, e in zip(filtering_predicates, inference_engines_required_predicates):
        if len(inference_services_executed.intersection(e)) == len(e):
          filtering_predicates_satisfied.append(p)

      where_str = get_where_str(filtering_predicates_satisfied)
      for k, v in original_column_mapping.items():
        where_str = where_str.replace(k, v)

      inp_query_str = f'''
        SELECT {inp_cols_str}
        {join_str}
        WHERE {where_str};
      '''

      async with self._sql_engine.begin() as conn:
        inp_df = await conn.run_sync(lambda conn: pd.read_sql(text(inp_query_str), conn))

      # The bound inference service is responsible for populating the database
      await bound_service.infer(inp_df)

      inference_services_executed.add(bound_service.service.name)

    # Execute the final query, now that all data is inserted
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.get_sql_query_text()))
      return res.fetchall()
