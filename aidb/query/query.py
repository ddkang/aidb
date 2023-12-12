from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import Dict

import sqlglot.expressions as exp
from sqlglot.rewriter import Rewriter
from sqlglot import Parser, Tokenizer, parse_one
from sympy import sympify
from sympy.logic.boolalg import to_cnf

from aidb.config.config import Config


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


def _remove_where(node):
  if isinstance(node, exp.Where):
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
  def all_queries_in_expressions(self):
    '''
    This function is used to extract all queries in the expression, including the entire query and subqueries
    '''
    all_queries = []
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Select):
        depth = 0
        parent = node.parent
        while parent:
          depth += 1
          parent = parent.parent

        extracted_query = Query(node.sql(), self.config)
        if depth != 0 :
          if extracted_query.is_approx_agg_query or extracted_query.is_approx_select_query:
            raise Exception("We don't support using approx query as a subquery")

        all_queries.append((Query(node.sql(), self.config), depth))

    all_queries = sorted(all_queries, key=lambda x:x[1], reverse=True)
    return all_queries


  def _check_in_subquery(self, node):
    '''
    this function check if current node is within a subquery
    '''
    node_parent = node
    while node_parent and not isinstance(node_parent, exp.Select):
      node_parent = node_parent.parent
    if node_parent is None or node_parent.parent is None:
      return False
    else:
      return True


  @cached_property
  def table_and_column_aliases_in_query(self):
    """
    finds the mapping of alias and original name presenting in the query, excluding subquery part.

    for e.g. for the query:
    SELECT t2.id AS col1, t2.frame AS col2, t2.column5
    FROM table_2 t2 JOIN blob_ids b
    ON t2.frame = b.frame
    WHERE b.frame > 102 and column1 > 950

    table_alias_to_name will be {b: blob_ids, t2: table_2}
    column_alias_to_name will be {col1: id, col2: frame}

    :return: mapping of table names and alias
    """
    table_alias_to_name = {}
    column_alias_to_name = {}
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Expression) and self._check_in_subquery(node):
        continue
      if isinstance(node, exp.Alias) and 'alias' in node.args and 'this' in node.args:
        if isinstance(node.args['this'], exp.Table):
          tbl_alias = node.args['alias'].args['this']
          if str.lower(tbl_alias) in table_alias_to_name:
            raise Exception('Duplicated alias found in query, please use another alias')
          tbl_name = node.args['this'].args['this'].args['this']
          table_alias_to_name[str.lower(tbl_alias)] = str.lower(tbl_name)

        elif isinstance(node.args['this'], exp.Column):
          col_alias = node.args['alias'].args['this']
          if str.lower(col_alias) in column_alias_to_name:
            raise Exception('Duplicated alias found in query, please use another alias')
          col_name = node.args['this'].args['this'].args['this']
          column_alias_to_name[str.lower(col_alias)] = str.lower(col_name)


    return table_alias_to_name, column_alias_to_name


  @cached_property
  def columns_in_query(self):
    """
    return the normalized column names in query, excluding subquery part
    """
    column_set = self._get_normalized_column_set(self.query_after_normalizing.get_expression())
    return column_set


  @cached_property
  def tables_in_query(self):
    table_set = set()
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Expression) and self._check_in_subquery(node):
        continue
      if isinstance(node, exp.Table):
        table_set.add(node.args['this'].args['this'])

    return list(table_set)


  def _get_predicate_name(self, predicate_count):
    predicate_name = f"P{predicate_count}"
    return predicate_name


  def _get_sympify_form(self, node, predicate_count, predicate_mappings):
    if node is None:
      return None, predicate_count
    elif isinstance(node, exp.Paren) or isinstance(node, exp.Where):
      assert "this" in node.args
      return self._get_sympify_form(node.args["this"], predicate_count, predicate_mappings)
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
            isinstance(node, exp.NEQ) or \
            isinstance(node, exp.In) or \
            isinstance(node, exp.Not) or \
            isinstance(node, exp.Column):
      predicate_name = self._get_predicate_name(predicate_count)
      predicate_mappings[predicate_name] = node
      return predicate_name, predicate_count + 1
    else:
      raise NotImplementedError


  def _get_or_clause_representation(self, or_expression, predicate_mappings):
    connected_by_ors = list(or_expression.args)
    predicates_in_ors = []
    if len(connected_by_ors) <= 1:
      predicates_in_ors.append(predicate_mappings[str(or_expression)])
    else:
      for s in connected_by_ors:
        predicates_in_ors.append(predicate_mappings[str(s)])
    return predicates_in_ors


  def _get_filtering_predicate_cnf_representation(self, cnf_expression, predicate_mappings):
    if '&' not in str(cnf_expression):
      return [self._get_or_clause_representation(cnf_expression, predicate_mappings)]

    or_expressions_connected_by_ands = list(cnf_expression.args)
    or_expressions_connected_by_ands_repr = []
    for or_expression in or_expressions_connected_by_ands:
      connected_by_ors = self._get_or_clause_representation(or_expression, predicate_mappings)
      or_expressions_connected_by_ands_repr.append(connected_by_ors)
    return or_expressions_connected_by_ands_repr


  def _replace_col_in_filter_predicate_with_root_col(self, expression):
    '''
    This function replace column with root columns,
    for e.g. in the filtering predicate 'colors.frame > 10000'
    the root column of 'colors.frame' is 'blob.frame'
    so filtering predicate will be converted into 'blob.frame > 10000'
    '''
    copied_expression = expression.copy()
    for node, _, _ in copied_expression.walk():
      if isinstance(node, exp.Expression) and self._check_in_subquery(node):
        continue
      if isinstance(node, exp.Column):
        table_name = str.lower(node.args['table'].args['this'])
        col_name = str.lower(node.args['this'].args['this'])
        normalized_col_name = f'{table_name}.{col_name}'
        originated_from = self.config.columns_to_root_column.get(normalized_col_name, normalized_col_name)
        table_name = originated_from.split('.')[0]
        node.args['table'].set('this', table_name)
    return copied_expression


  @cached_property
  def filtering_predicates(self):
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

    (blobs.timestamp > 10 or colors.color = red) and (blobs.timestamp < 12 or colors.color = red)
    """
    normalized_exp = self.query_after_normalizing.get_expression()
    if normalized_exp.find(exp.Where) is not None:
      if self._tables is None:
        raise Exception("Need table and column information to support alias")
      # predicate name (used for sympy package) to expression
      predicate_mappings = {}
      # predicate mapping will be filled by this function
      sympy_representation, _ = self._get_sympify_form(normalized_exp.find(exp.Where), 0, predicate_mappings)
      sympy_expression = sympify(sympy_representation)
      cnf_expression = to_cnf(sympy_expression)
      filtering_predicates = self._get_filtering_predicate_cnf_representation(cnf_expression, predicate_mappings)
      filtering_clauses = []
      for or_connected_filtering_predicate in filtering_predicates:
        or_connected_clauses = []
        for fp in or_connected_filtering_predicate:
          new_fp = self._replace_col_in_filter_predicate_with_root_col(fp)
          or_connected_clauses.append(new_fp)
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

  @cached_property
  def query_after_normalizing(self):
    """
    this function traverse expression and return the normalized query excluding subquery part.
    Specifically, normalizing expression includes removing alias, adding affiliate table name for columns, replacing
    alias with original name
    e.g. for this query: SELECT frame FROM blobs b WHERE b.timestamp > 100
    the expression will be converted into SELECT blobs.frame FROM blobs WHERE blobs.timestamp > 100
    """

    def _remove_alias_in_expression(original_list):
      removed_alias_list = []
      for element in original_list:
        if isinstance(element, exp.Alias):
          removed_alias_list.append(element.args['this'])
        else:
          removed_alias_list.append(element)
      return removed_alias_list


    copied_expression = self._expression.copy()
    table_alias_to_name, column_alias_to_name = self.table_and_column_aliases_in_query
    for node, _, _ in copied_expression.walk():
      if isinstance(node, exp.Expression) and self._check_in_subquery(node):
        continue
      if isinstance(node, exp.Column):
        if isinstance(node.args['this'], exp.Identifier):
          col_name = str.lower(node.args['this'].args['this'])
          if col_name in column_alias_to_name:
            col_name = column_alias_to_name[col_name]
            node.args['this'].set('this', col_name)
          if 'table' in node.args and node.args['table'] is not None:
            table_name = str.lower(node.args['table'].args['this'])
            if table_name in table_alias_to_name:
              table_name = table_alias_to_name[table_name]
          else:
            table_name = self._get_table_of_column(col_name)

          node.set('table', exp.Identifier(this=table_name, quoted=False))

    # remove alias

    copied_expression.set('expressions', _remove_alias_in_expression(copied_expression.args['expressions']))
    copied_expression.args['from'].set(
      'expressions',
      _remove_alias_in_expression(copied_expression.args['from'].args['expressions'])
    )

    for join_element in copied_expression.args['joins']:
      if isinstance(join_element.args['this'], exp.Alias):
        join_element.set('this', join_element.args['this'].args['this'])

    return Query(copied_expression.sql(), self.config)


  def _get_normalized_column_set(self, normalized_expression):
    """
      this function assume the input expression has been normalized, the function traverses normalized expression
      and return the list of normalized column names excluding subquery part
      e.g. for this query: SELECT blobs.frame FROM blobs WHERE blobs.timestamp > 100
      it will return [blobs.frame, blobs.timestamp],
      """
    normalized_column_set = set()
    for node, _, _ in normalized_expression.walk():
      if isinstance(node, exp.Expression) and self._check_in_subquery(node):
        continue
      if isinstance(node, exp.Column):
        if isinstance(node.args['this'], exp.Identifier):
          col_name = node.args['this'].args['this']
          table_name = node.args['table'].args['this']
          normalized_column_set.add(f'{table_name}.{col_name}')
        elif isinstance(node.args['this'], exp.Star):
          for table_name in self.tables_in_query:
            for col_name, _ in self._tables[table_name].items():
              normalized_column_set.add(f'{table_name}.{col_name}')
    return normalized_column_set


  @cached_property
  def inference_engines_required_for_filtering_predicates(self):
    """
    Inference services required to run to satisfy the columns present in each filtering predicate
    for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
    it returns [{colors02}, [], {objects00}]
    """
    inference_engines_required_predicates = []
    for filtering_predicate in self.filtering_predicates:
      inference_engines_required = set()
      for or_connected_predicate in filtering_predicate:
        normalized_col_set = self._get_normalized_column_set(or_connected_predicate)
        for normalized_col_name in normalized_col_set:
          if normalized_col_name in self.config.column_by_service:
            inference_engines_required.add(self.config.column_by_service[normalized_col_name].service.name)
      inference_engines_required_predicates.append(inference_engines_required)
    return inference_engines_required_predicates


  @cached_property
  def tables_in_filtering_predicates(self):
    """
    Tables needed to satisfy the columns present in each filtering predicate
    for e.g. if predicates are [[color=red],[frame>20],[object_class=car]]
    it returns [{color}, {blob}, {object}]
    """
    tables_required_predicates = []
    for filtering_predicate in self.filtering_predicates:
      tables_required = set()
      for or_connected_predicate in filtering_predicate:
        normalized_col_set = self._get_normalized_column_set(or_connected_predicate)
        for normalized_col_name in normalized_col_set:
          if normalized_col_name in self.config.column_by_service:
            tables_required.add(normalized_col_name.split('.')[0])
      tables_required_predicates.append(tables_required)
    return tables_required_predicates


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
    query_no_aqp_expression = self.base_sql_no_aqp.get_expression()
    if not isinstance(query_no_aqp_expression, exp.Select):
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
    for expression in query_no_aqp_expression.args['expressions']:
      expression_counts[type(expression)] += 1

    if len(expression_counts) > 1:
      raise Exception('Multiple expression types found')

    if exp.Avg not in expression_counts and exp.Count not in expression_counts and exp.Sum not in expression_counts:
      raise Exception('We only support approximation for Avg, Sum and Count query currently.')

    if not self.error_target or not self.confidence:
      raise Exception('Aggregation query should contain error target and confidence')

    return True


  @cached_property
  def recall_target(self):
    recall_target = self._get_keyword_arg(exp.RecallTarget)
    return recall_target / 100. if recall_target else None


  @cached_property
  def precision_target(self):
    precision_target = self._get_keyword_arg(exp.PrecisionTarget)
    return precision_target / 100. if precision_target else None


  @cached_property
  def oracle_budget(self):
    oracle_budget = self._get_keyword_arg(exp.Budget)
    return oracle_budget if oracle_budget else None


  @cached_property
  def is_approx_select_query(self):
    if self.precision_target is not None:
      raise Exception("We haven't support approx select query with precision target.")
    return self.recall_target is not None


  @cached_property
  def is_valid_approx_select_query(self):
    if self.oracle_budget or self.confidence is None:
      return False
    else:
      return True


  @cached_property
  def base_sql_no_aqp(self):
    _exp_no_aqp = self._expression.transform(_remove_aqp)
    if _exp_no_aqp is None:
      raise Exception('SQL contains no non-AQP statements')
    return Query(_exp_no_aqp.sql(), self.config)


  @cached_property
  def base_sql_no_where(self):
    _exp_no_where = self._expression.transform(_remove_where)
    if _exp_no_where is None:
      raise Exception('SQL contains no non-AQP statements')
    return Query(_exp_no_where.sql(), self.config)


  # Get aggregation type
  @cached_property
  def get_agg_type(self):
    # Only support one aggregation for the time being
    query_no_aqp_expression = self.base_sql_no_aqp.get_expression()
    if len(query_no_aqp_expression.args['expressions']) != 1:
      raise Exception('Multiple expressions found')
    select_exp = query_no_aqp_expression.args['expressions'][0]
    if isinstance(select_exp, exp.Avg):
      return exp.Avg
    elif isinstance(select_exp, exp.Sum):
      return exp.Sum
    elif isinstance(select_exp, exp.Count):
      return exp.Count
    else:
      raise Exception('Unsupported aggregation')


  # FIXME: move it to sqlglot.rewriter
  def add_where_condition(self, operator:str , where_condition):
    expression = self._expression.copy()
    re = Rewriter(expression)
    new_sql = re.add_where(operator, where_condition)
    return Query(new_sql.expression.sql(), self.config)


  def add_select(self, selects):
    expression = self._expression.copy()
    re = Rewriter(expression)
    new_sql = re.add_selects(selects)
    return Query(new_sql.expression.sql(), self.config)


  def add_join(self, new_join):
    expression = self._expression.copy()
    re = Rewriter(expression)
    new_sql = re.add_join(new_join)
    return Query(new_sql.expression.sql(), self.config)
