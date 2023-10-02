from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

import sqlglot.expressions as exp
from aidb.config.config import Config
from aidb.query.utils import (Expression, FilteringClause, FilteringPredicate,
                              change_literal_type_to_col_type,
                              extract_column_or_value)
from sqlglot import Parser, Tokenizer
from sympy import sympify
from sympy.logic.boolalg import to_cnf


@dataclass
class Query(object):
  """
  class to hold sql query related data
  this is immutable class
  """
  sql_str: str
  config: Config

  @cached_property
  def _tokens(self):
    return Tokenizer().tokenize(self.sql_str)

  @cached_property
  def _expression(self) -> exp.Expression:
    return Parser().parse(self._tokens)[0]

  @cached_property
  def sql_query_text(self):
    return self._expression.sql()

  @cached_property
  def _tables(self) -> Dict[str, Dict[str, str]]:
    """
    dictionary of tables to columns and their types
    """
    _tables = dict()
    for table_name, table in self.config.tables.items():
      _tables[table_name] = dict()
      for column in table.columns:
        _tables[table_name][column.name] = column.type
    return _tables

  @cached_property
  def table_name_to_aliases(self):
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

  @cached_property
  def table_aliases_to_name(self):
    table_name_to_alias = self.table_name_to_aliases
    return {v: k for k, v in table_name_to_alias.items()}

  @cached_property
  def tables_in_query(self):
    table_list = set()
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Table):
        table_list.add(node.args["this"].args["this"])
    return table_list

  def _get_predicate_name(self):
    self._predicate_count += 1
    predicate_name = f"P{self._predicate_count}"
    return predicate_name

  def _get_sympify_form(self, node):
    if node is None:
      return None
    elif isinstance(node, exp.Paren) or isinstance(node, exp.Where):
      assert "this" in node.args
      return self._get_sympify_form(node.args["this"])
    elif isinstance(node, exp.Column):
      predicate_name = self._get_predicate_name()
      self._predicate_mappings[predicate_name] = node
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
      self._predicate_mappings[predicate_name] = node
      return predicate_name
    else:
      raise NotImplementedError

  def _get_original_predicate(self, predicate_name) -> FilteringPredicate:
    if predicate_name[0] == "~":
      return FilteringPredicate(True, self._predicate_mappings[predicate_name[1:]])
    else:
      return FilteringPredicate(False, self._predicate_mappings[predicate_name])

  def _get_or_clause_representation(self, or_expression) -> List[FilteringPredicate]:
    connected_by_ors = list(or_expression.args)
    predicates_in_ors = []
    if len(connected_by_ors) <= 1:
      predicates_in_ors.append(self._get_original_predicate(str(or_expression)))
    else:
      for s in connected_by_ors:
        predicates_in_ors.append(self._get_original_predicate(str(s)))
    return predicates_in_ors

  def _get_filtering_predicate_cnf_representation(self, cnf_expression) -> List[List[FilteringPredicate]]:
    if '&' not in str(cnf_expression):
      return [self._get_or_clause_representation(cnf_expression)]

    or_expressions_connected_by_ands = list(cnf_expression.args)
    or_expressions_connected_by_ands_repr = []
    for or_expression in or_expressions_connected_by_ands:
      connected_by_ors = self._get_or_clause_representation(or_expression)
      or_expressions_connected_by_ands_repr.append(connected_by_ors)
    return or_expressions_connected_by_ands_repr

  @cached_property
  def filtering_predicates(self) -> List[List[FilteringClause]]:
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
    if self._expression.find(exp.Where) is not None:
      if self._tables is None:
        raise Exception("Need table and column information to support alias")
      self._predicate_count = 0
      # predicate name (used for sympy package) to expression
      self._predicate_mappings = {}
      sympy_representation = self._get_sympify_form(self._expression.find(exp.Where))
      sympy_expression = sympify(sympy_representation)
      cnf_expression = to_cnf(sympy_expression)
      filtering_predicates = self._get_filtering_predicate_cnf_representation(cnf_expression)
      filtering_clauses = []
      for or_connected_filtering_predicate in filtering_predicates:
        or_connected_clauses = []
        for fp in or_connected_filtering_predicate:
          if isinstance(fp.predicate, exp.Column):
            # in case of boolean type columns
            column_name = self._get_normalized_col_name_from_col_exp(fp.predicate)
            or_connected_clauses.append(
              FilteringClause(fp.is_negation, exp.Column, Expression("column", column_name), None))
          else:
            t1, v1 = extract_column_or_value(fp.predicate.args["this"])
            t2, v2 = extract_column_or_value(fp.predicate.args["expression"])
            if t1 == "column":
              left_value = self._get_normalized_col_name_from_col_exp(v1)
            else:
              left_value = v1
            if t2 == "column":
              right_value = self._get_normalized_col_name_from_col_exp(v2)
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
      return []

  def _get_table_of_column(self, col_name):
    tables_of_column = []
    for table in self.tables_in_query:
      if col_name in self._tables[table]:
        tables_of_column.append(table)
    if len(tables_of_column) == 0:
      raise Exception(f"Column - {col_name} is not present in any table")
    elif len(tables_of_column) > 1:
      raise Exception(f"Ambiguity in identifying column - {col_name}, it is present in multiple tables")
    else:
      return tables_of_column[0]

  def _get_normalized_col_name_from_col_exp(self, node):
    """
    this function uses tables (and their alias) and the columns present in them.
    to return the column name in the form {table_name}.{col_name}
    e.g. for this query: SELECT frame FROM blobs b WHERE b.timestamp > 100
    for 'frame' column, it will return 'blobs.frame' and for 'timestamp' it will return 'blobs.timestamp'
    """
    if "table" in node.args and node.args["table"] is not None:
      table_name = str.lower(node.args["table"].args["this"])
      if table_name in self.table_aliases_to_name:
        table_name = self.table_aliases_to_name[table_name]
    else:
      table_name = self._get_table_of_column(node.args["this"].args["this"])
    return f"{table_name}.{node.args['this'].args['this']}"

  @cached_property
  def inference_engines_required_for_filtering_predicates(self):
    """
    Inference services required to run to satisfy the columns present in each filtering predicate
    for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
    it returns [[color], [], [object]]
    """
    inference_engines_required_predicates = []
    for filtering_predicate in self.filtering_predicates:
      inference_engines_required = set()
      for or_connected_predicate in filtering_predicate:
        if or_connected_predicate.left_exp.type == "column":
          originated_from = self.config.columns_to_root_column.get(or_connected_predicate.left_exp.value,
                                                                   or_connected_predicate.left_exp.value)
          if originated_from in self.config.column_by_service:
            inference_engines_required.add(self.config.column_by_service[originated_from].service.name)
        if or_connected_predicate.right_exp.type == "column":
          originated_from = self.config.columns_to_root_column.get(or_connected_predicate.right_exp.value,
                                                                   or_connected_predicate.right_exp.value)
          if originated_from in self.config.column_by_service:
            inference_engines_required.add(self.config.column_by_service[originated_from].service.name)
      inference_engines_required_predicates.append(inference_engines_required)
    return inference_engines_required_predicates

  @cached_property
  def tables_in_filtering_predicates(self):
    """
    Tables needed to satisfy the columns present in each filtering predicate
    for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
    it returns [[color], [blob], [object]]
    """
    tables_required_predicates = []
    for filtering_predicate in self.filtering_predicates:
      tables_required = set()
      for or_connected_predicate in filtering_predicate:
        if or_connected_predicate.left_exp.type == "column":
          originated_from = self.config.columns_to_root_column.get(or_connected_predicate.left_exp.value,
                                                                   or_connected_predicate.left_exp.value)
          tables_required.add(originated_from.split('.')[0])
        if or_connected_predicate.right_exp.type == "column":
          originated_from = self.config.columns_to_root_column.get(or_connected_predicate.right_exp.value,
                                                                   or_connected_predicate.right_exp.value)
          tables_required.add(originated_from.split('.')[0])
      tables_required_predicates.append(tables_required)
    return tables_required_predicates
