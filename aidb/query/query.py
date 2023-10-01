import sqlglot.expressions as exp
from sqlglot import Tokenizer, Parser

from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List, Tuple

from aidb.config.config_types import Table
from aidb.query.utils import change_literal_type_to_col_type, FilteringClause, Expression

import sqlglot.expressions as exp
from sympy.logic.boolalg import to_cnf
from sympy import sympify
from aidb.query.utils import FilteringPredicate


@dataclass
class Query(object):
  sql_str: str
  tables: Dict[str, Table]

  @cached_property
  def _tokens(self):
    return Tokenizer().tokenize(self.sql_str)

  @cached_property
  def _expression(self) -> exp.Expression:
    return Parser().parse(self._tokens)[0]

  @cached_property
  def _tables(self):
    _tables = dict()
    for table_name, table in self.tables.items():
      _tables[table_name] = dict()
      for column in table.columns:
        _tables[table_name][column.name] = column.type
    return _tables

  def _get_predicate_name(self):
    self.predicate_count += 1
    predicate_name = f"P{self.predicate_count}"
    return predicate_name

  def _get_sympify_form(self, node):
    if node is None:
      return None
    elif isinstance(node, exp.Paren) or isinstance(node, exp.Where):
      assert "this" in node.args
      return self._get_sympify_form(node.args["this"])
    elif isinstance(node, exp.Column):
      predicate_name = self._get_predicate_name()
      self.predicate_mappings[predicate_name] = node
      return predicate_name
    elif isinstance(node, exp.Not):
      return f"~({self._get_sympify_form(node.args['this'])})"
    elif isinstance(node, exp.And):
      assert "this" in node.args and "expression" in node.args
      return f"({self._get_sympify_form(node.args['this'])} & " \
             f"{self._get_sympify_form(node.args['expression'])})"
    elif isinstance(node, exp.Or):
      assert "this" in node.args and "expression" in node.args
      return f"({self._get_sympify_form(node.args['this'])} | " \
             f"{self._get_sympify_form(node.args['expression'])})"
    elif isinstance(node, exp.GT) or \
            isinstance(node, exp.LT) or \
            isinstance(node, exp.GTE) or \
            isinstance(node, exp.LTE) or \
            isinstance(node, exp.EQ) or \
            isinstance(node, exp.Like) or \
            isinstance(node, exp.NEQ):
      # TODO: chained comparison operators not supported
      assert "this" in node.args and "expression" in node.args
      predicate_name = self._get_predicate_name()
      self.predicate_mappings[predicate_name] = node
      return predicate_name
    else:
      raise NotImplementedError

  def get_cnf_form_predicate(self):
    return self.cnf_expression

  def _get_original_predicate(self, predicate_name):
    if predicate_name[0] == "~":
      return FilteringPredicate(True, self.predicate_mappings[predicate_name[1:]])
    else:
      return FilteringPredicate(False, self.predicate_mappings[predicate_name])

  def _get_or_clause_representation(self, or_expression):
    connected_by_ors = list(or_expression.args)
    predicates_in_ors = []
    if len(connected_by_ors) <= 1:
      predicates_in_ors.append(self._get_original_predicate(str(or_expression)))
    else:
      for s in connected_by_ors:
        predicates_in_ors.append(self._get_original_predicate(str(s)))
    return predicates_in_ors

  def _get_filtering_predicate_cnf_representation(self):
    if '&' not in str(self.cnf_expression):
      return [self._get_or_clause_representation(self.cnf_expression)]

    or_expressions_connected_by_ands = list(self.cnf_expression.args)
    or_expressions_connected_by_ands_repr = []
    for or_expression in or_expressions_connected_by_ands:
      connected_by_ors = self._get_or_clause_representation(or_expression)
      or_expressions_connected_by_ands_repr.append(connected_by_ors)
    return or_expressions_connected_by_ands_repr

  @cached_property
  def filtering_predicates(self) -> List[List[FilteringClause]]:
    if self._expression.find(exp.Where) is not None:
      self.predicate_count = 0
      # predicate name (used for sympy package) to expression
      self.predicate_mappings = {}
      self.sympy_representation = self._get_sympify_form(self._expression.find(exp.Where))
      self.sympy_expression = sympify(self.sympy_representation)
      self.cnf_expression = to_cnf(self.sympy_expression)
      return self.get_filtering_predicate_cnf_representation()
    else:
      return []

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
    """
      this class contains the functionality to convert filtering predicate in SQL query to
      conjunctive normal form.
      for e.g. in the following query

      SELECT c.color , COUNT(c.blob_id)
      FROM Colors c JOIN Blobs b
      ON c.blob_id = b.blob_id
      WHERE (b.timestamp > 10 and b.timestamp < 12) or c.color=red
      GROUP BY c.blob_id

      the filtering predicate will be converted to

      (b.timestamp > 10 or c.color = red) and (b.timestamp < 12 or c.color = red)
    """
    filtering_predicates = self._get_filtering_predicate_cnf_representation()
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