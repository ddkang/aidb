import sqlglot.expressions as exp
from sympy.logic.boolalg import to_cnf
from sympy import sympify
from aidb.query.utils import FilteringPredicate


class QueryCNFUtil:
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

  def __init__(self, where_exp: exp):
    self.predicate_count = 0
    # predicate name (used for sympy package) to expression
    self.predicate_mappings = {}
    self.sympy_representation = self._get_sympify_form(where_exp)
    self.sympy_expression = sympify(self.sympy_representation)
    self.cnf_expression = to_cnf(self.sympy_expression)

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

  def get_filtering_predicate_cnf_representation(self):
    if '&' not in str(self.cnf_expression):
      return [self._get_or_clause_representation(self.cnf_expression)]

    or_expressions_connected_by_ands = list(self.cnf_expression.args)
    or_expressions_connected_by_ands_repr = []
    for or_expression in or_expressions_connected_by_ands:
      connected_by_ors = self._get_or_clause_representation(or_expression)
      or_expressions_connected_by_ands_repr.append(connected_by_ors)
    return or_expressions_connected_by_ands_repr
