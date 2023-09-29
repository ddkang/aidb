from __future__ import annotations

import sqlglot.expressions as exp
from sqlglot import Tokenizer, Parser
from typing import List

from aidb.query.cnf_query_utils import QueryCNFUtil
from aidb.utils.logger import logger
from functools import cached_property

from aidb.query.utils import change_literal_type_to_col_type, FilteringClause, Expression


class Query(object):
  def __init__(self, sql, tables):
    self._sql = sql
    self._tokens = Tokenizer().tokenize(sql)
    self._expression: exp.Expression = Parser().parse(self._tokens)[0]
    self._tables = dict()
    for table_name, table in tables.items():
      self._tables[table_name] = dict()
      for column in table.columns:
        self._tables[table_name][column.name] = column.type

  @cached_property
  def filtering_predicates(self) -> List[List[FilteringClause]]:
    return self.get_filtering_predicate_cnf_representation()

  def _get_table_aliases(self):
    """
    finds the mapping of table names and aliases present in the query.

    for e.g. for the query:
    SELECT t2.id, t2.frame, t2.column5
    FROM table_2 t2 JOIN blob_ids b
    ON t2.frame = b.frame
    WHERE b.frame > 102 and column1 > 950

    it will return {blob_ids: b, table_2: t2}

    :return: mapping of table names and alias
    """
    table_alias = {}
    for alias_exp in self._expression.find_all(exp.Alias):
      for node, _, _ in alias_exp.walk():
        if isinstance(node, exp.Alias) and "alias" in node.args and "this" in node.args:
          if isinstance(node.args["this"], exp.Table):
            tbl_name = node.args["this"].args["this"].args["this"]
            tbl_alias = node.args["alias"].args["this"]
            table_alias[str.lower(tbl_name)] = str.lower(tbl_alias)
    return table_alias

  def _get_table_of_column(self, col_name, tables_info, tables_in_query):
    tables_of_column = []
    for table in tables_in_query:
      if col_name in tables_info[table]:
        tables_of_column.append(table)
    if len(tables_of_column) == 0:
      raise Exception(f"Column - {col_name} is not present in any table")
    elif len(tables_of_column) > 1:
      raise Exception(f"Ambiguity in identifying column - {col_name}, it is present in multiple tables")
    else:
      return tables_of_column[0]

  def _get_tables_in_query(self):
    table_list = set()
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Table):
        table_list.add(node.args["this"].args["this"])
    return table_list

  def _get_normalized_col_name_from_col_exp(self, node, table_alias_to_name, tables_info, tables_present_in_query):
    """
    this function uses tables (and their alias) and the columns present in them.
    to return the column name in the form {table_name}.{col_name}
    e.g. for this query: SELECT frame FROM blobs b WHERE b.timestamp > 100
    for 'frame' column, it will return 'blobs.frame' and for 'timestamp' it will return 'blobs.timestamp'
    """
    if "table" in node.args and node.args["table"] is not None:
      table_name = str.lower(node.args["table"].args["this"])
      if table_name in table_alias_to_name:
        table_name = table_alias_to_name[table_name]
    else:
      table_name = self._get_table_of_column(node.args["this"].args["this"], tables_info, tables_present_in_query)
    return f"{table_name}.{node.args['this'].args['this']}"

  def get_sql_query_text(self):
    return self._expression.sql()

  def _extract_column_or_value(self, node):
    """
    return a tuple containing if this is column or literal
    :param node:
    :return:
    """
    if isinstance(node, exp.Paren):
      return self._extract_column_or_value(node.args["this"])
    elif isinstance(node, exp.Column):
      return ("column", node)
    elif isinstance(node, exp.Literal) or isinstance(node, exp.Boolean):
      return ("literal", node.args["this"])
    else:
      raise Exception("Not Supported Yet")

  def get_filtering_predicate_cnf_representation(self):
    if self._expression.find(exp.Where) is not None:
      cnf_query = QueryCNFUtil(self._expression.find(exp.Where))
      filtering_predicates = cnf_query.get_filtering_predicate_cnf_representation()
      if self._tables is None:
        raise Exception("Need table and column information to support alias")
      filtering_clauses = []
      table_name_to_alias = self._get_table_aliases()
      table_alias_to_name = {v: k for k, v in table_name_to_alias.items()}
      tables_present_in_query = self._get_tables_in_query()
      for or_connected_filtering_predicate in filtering_predicates:
        or_connected_clauses = []
        for fp in or_connected_filtering_predicate:
          if isinstance(fp.predicate, exp.Column):
            # in case of boolean type columns
            column_name = self._get_normalized_col_name_from_col_exp(
              fp.predicate, table_alias_to_name, self._tables, tables_present_in_query)
            or_connected_clauses.append(
              FilteringClause(fp.is_negation, exp.Column, Expression("column", column_name), None))
          else:
            t1, v1 = self._extract_column_or_value(fp.predicate.args["this"])
            t2, v2 = self._extract_column_or_value(fp.predicate.args["expression"])
            if t1 == "column":
              left_value = self._get_normalized_col_name_from_col_exp(
                v1, table_alias_to_name, self._tables, tables_present_in_query)
            else:
              left_value = v1
            if t2 == "column":
              right_value = self._get_normalized_col_name_from_col_exp(
                v2, table_alias_to_name, self._tables, tables_present_in_query)
            else:
              right_value = v2

            if t1 == "literal" and t2 == "column":
              t_name, c_name = right_value.split('.')
              left_value = change_literal_type_to_col_type(self._tables[t_name][c_name], left_value)
              # change compare_value_2 to float or int
            elif t2 == "literal" and t1 == "column":
              t_name, c_name = left_value.split('.')
              right_value = change_literal_type_to_col_type(self._tables[t_name][c_name], right_value)
            elif t1 == "literal" and t2 == "literal":
              # both left and right cannot be literals
              raise Exception("Comparisons among literals not supported in filtering predicate")

            or_connected_clauses.append(
              FilteringClause(fp.is_negation, type(fp.predicate), Expression(t1, left_value),
                              Expression(t2, right_value)))
        filtering_clauses.append(or_connected_clauses)
      return filtering_clauses
    else:
      logger.info('No Filtering Predicate')
      return []
