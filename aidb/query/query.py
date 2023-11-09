import copy
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, List

import sqlglot.expressions as exp
from sqlglot.rewriter import Rewriter
from sqlglot import Parser, Tokenizer, parse_one
from sympy import sympify
from sympy.logic.boolalg import to_cnf

from aidb.config.config import Config
from aidb.query.utils import (Expression, FilteringClause, FilteringPredicate,
                              change_literal_type_to_col_type,
                              extract_column_or_value)


def is_aqp_exp(node):
  return isinstance(node, exp.ErrorTarget) \
      or isinstance(node, exp.RecallTarget) \
      or isinstance(node, exp.PrecisionTarget) \
      or isinstance(node, exp.Confidence) \
      or isinstance(node, exp.Budget)


def _remove_aqp(node):
  if is_aqp_exp(node):
    return None
  return node


@dataclass(frozen=True)
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


  def get_expression(self):
    return self._expression


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


  def _get_predicate_name(self, predicate_count):
    predicate_name = f"P{predicate_count}"
    return predicate_name


  def _get_sympify_form(self, node, predicate_count, predicate_mappings):
    if node is None:
      return None, predicate_count
    elif isinstance(node, exp.Paren) or isinstance(node, exp.Where):
      assert "this" in node.args
      return self._get_sympify_form(node.args["this"], predicate_count, predicate_mappings)
    elif isinstance(node, exp.Column):
      predicate_name = self._get_predicate_name(predicate_count)
      predicate_mappings[predicate_name] = node
      return predicate_name, predicate_count + 1
    elif isinstance(node, exp.Not):
      e1, p1 = self._get_sympify_form(node.args['this'], predicate_count, predicate_mappings)
      return f"~({e1})", p1
    elif isinstance(node, exp.And):
      assert "this" in node.args and "expression" in node.args
      e1, p1 = self._get_sympify_form(node.args['this'], predicate_count, predicate_mappings)
      e2, p2 = self._get_sympify_form(node.args['expression'], p1, predicate_mappings)
      return f"({e1} & {e2})", p2
    elif isinstance(node, exp.Or):
      assert "this" in node.args and "expression" in node.args
      e1, p1 = self._get_sympify_form(node.args['this'], predicate_count, predicate_mappings)
      e2, p2 = self._get_sympify_form(node.args['expression'], p1, predicate_mappings)
      return f"({e1} | {e2})", p2
    elif isinstance(node, exp.GT) or \
            isinstance(node, exp.LT) or \
            isinstance(node, exp.GTE) or \
            isinstance(node, exp.LTE) or \
            isinstance(node, exp.EQ) or \
            isinstance(node, exp.Like) or \
            isinstance(node, exp.NEQ):
      # TODO: chained comparison operators not supported
      assert "this" in node.args and "expression" in node.args
      predicate_name = self._get_predicate_name(predicate_count)
      predicate_mappings[predicate_name] = node
      return predicate_name, predicate_count + 1
    else:
      raise NotImplementedError


  def _get_original_predicate(self, predicate_name, predicate_mappings) -> FilteringPredicate:
    if predicate_name[0] == "~":
      return FilteringPredicate(True, predicate_mappings[predicate_name[1:]])
    else:
      return FilteringPredicate(False, predicate_mappings[predicate_name])


  def _get_or_clause_representation(self, or_expression, predicate_mappings) -> List[FilteringPredicate]:
    connected_by_ors = list(or_expression.args)
    predicates_in_ors = []
    if len(connected_by_ors) <= 1:
      predicates_in_ors.append(self._get_original_predicate(str(or_expression), predicate_mappings))
    else:
      for s in connected_by_ors:
        predicates_in_ors.append(self._get_original_predicate(str(s), predicate_mappings))
    return predicates_in_ors


  def _get_filtering_predicate_cnf_representation(self, cnf_expression, predicate_mappings) -> List[
    List[FilteringPredicate]]:
    if '&' not in str(cnf_expression):
      return [self._get_or_clause_representation(cnf_expression, predicate_mappings)]

    or_expressions_connected_by_ands = list(cnf_expression.args)
    or_expressions_connected_by_ands_repr = []
    for or_expression in or_expressions_connected_by_ands:
      connected_by_ors = self._get_or_clause_representation(or_expression, predicate_mappings)
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
      # predicate name (used for sympy package) to expression
      predicate_mappings = {}
      # predicate mapping will be filled by this function
      sympy_representation, _ = self._get_sympify_form(self._expression.find(exp.Where), 0, predicate_mappings)
      sympy_expression = sympify(sympy_representation)
      cnf_expression = to_cnf(sympy_expression)
      filtering_predicates = self._get_filtering_predicate_cnf_representation(cnf_expression, predicate_mappings)
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


  @cached_property
  def columns_in_query(self):
    """
    nested queries are not supported for the time being
    * is supported
    """
    column_set = set()
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Column):
        if isinstance(node.args['this'], exp.Identifier):
          column_set.add(self._get_normalized_col_name_from_col_exp(node))
        elif isinstance(node.args['this'], exp.Star):
          for table in self.tables_in_query:
            for col, _ in self._tables[table].items():
              column_set.add(f"{table}.{col}")
        else:
          raise Exception('Unsupported column type')
    return column_set


  @cached_property
  def inference_engines_required_for_query(self):
    """
    Inference services required for sql query, will return a list of inference service
    """
    visited = self.columns_in_query.copy()
    stack = list(visited)
    inference_engines_required = set()

    while stack:
      col = stack.pop()

      if col in self.config.column_by_service:
        inference = self.config.column_by_service[col]

        if inference not in inference_engines_required:
          inference_engines_required.add(inference)

          for inference_col in inference.binding.input_columns:
            if inference_col not in visited:
              stack.append(inference_col)
              visited.add(inference_col)

    inference_engines_ordered = [
      inference_engine
      for inference_engine in self.config.inference_topological_order
      if inference_engine in inference_engines_required
    ]

    return inference_engines_ordered


  @cached_property
  def blob_tables_required_for_query(self):
    '''
    get required blob tables for a query
    '''
    blob_tables = set()
    for inference_engine in self.inference_engines_required_for_query:
      for input_col in inference_engine.binding.input_columns:
        input_table = input_col.split('.')[0]
        if input_table in self.config.blob_tables:
          blob_tables.add(input_table)

    return list(blob_tables)


  def _get_keyword_arg(self, exp_type):
    value = None
    for node, _, key in self._expression.walk():
      if isinstance(node, exp_type):
        if value is not None:
          raise Exception(f'Multiple unexpected keywords found')
        else:
          value = float(node.args['this'].args['this'])
    return value


  @cached_property
  def limit_cardinality(self):
    return self._get_keyword_arg(exp.Limit)


  def is_limit_query(self):
    cardinality = self.limit_cardinality
    if cardinality is None:
      return False
    return True


  @cached_property
  def error_target(self):
    error_target = self._get_keyword_arg(exp.ErrorTarget)
    return error_target / 100. if error_target else None


  @cached_property
  def confidence(self):
    return self._get_keyword_arg(exp.Confidence)


  @cached_property
  def is_approx_agg_query(self):
    return self.error_target is not None and self.confidence is not None


  def is_select_query(self):
    return isinstance(self._expression, exp.Select)


  # Validate AQP
  def is_valid_aqp_query(self):
    # Only accept select statements
    if not isinstance(self.base_sql_no_aqp, exp.Select):
      raise Exception('Not a select statement')

    if self._expression.find(exp.Group) is not None:
      raise Exception(
          '''We do not support GROUP BY for approximate aggregation queries. 
          Try running without the error target and confidence.'''
      )

    if self._expression.find(exp.Join) is not None:
      raise Exception(
          '''We do not support Join for approximate aggregation queries. 
          Try running without the error target and confidence.'''
      )

    # Count the number of distinct aggregates
    expression_counts = defaultdict(int)
    for expression in self.base_sql_no_aqp.args['expressions']:
      expression_counts[type(expression)] += 1

    if len(expression_counts) > 1:
      raise Exception('Multiple expression types found')

    if exp.Avg not in expression_counts and exp.Count not in expression_counts and exp.Sum not in expression_counts:
      raise Exception('We only support approximation for Avg, Sum and Count query currently.')

    if not self.error_target or not self.confidence:
      raise Exception('Aggregation query should contain error target and confidence')

    return True


  @cached_property
  def base_sql_no_aqp(self):
    _exp_no_aqp = self._expression.transform(_remove_aqp)
    if _exp_no_aqp is None:
      raise Exception('SQL contains no non-AQP statements')
    return _exp_no_aqp


  # Get aggregation type
  @cached_property
  def get_agg_type(self):
    # Only support one aggregation for the time being
    if len(self.base_sql_no_aqp.args['expressions']) != 1:
      raise Exception('Multiple expressions found')
    select_exp = self.base_sql_no_aqp.args['expressions'][0]
    if isinstance(select_exp, exp.Avg):
      return exp.Avg
    elif isinstance(select_exp, exp.Sum):
      return exp.Sum
    elif isinstance(select_exp, exp.Count):
      return exp.Count
    else:
      raise Exception('Unsupported aggregation')


  # FIXME: move it to sqlglot.rewriter
  def add_where_condition(self, expression, operator:str , where_condition):
    expression = copy.deepcopy(expression)
    where = expression.find(exp.Where)
    new_condition = parse_one(where_condition)
    if where:
      if operator.upper() == 'AND':
        where.args['this'] = exp.And(this=new_condition, expression=where.args['this'])
      elif operator.upper() == 'OR':
        where.args['this'] = exp.Or(this=new_condition, expression=where.args['this'])
      return expression
    else:
      where_expr = exp.Where(this=new_condition)
      expression.args['where'] = where_expr
      return expression


  def add_select(self, expression, selects):
    re = Rewriter(expression)
    e = re.add_selects(selects)
    return e.expression