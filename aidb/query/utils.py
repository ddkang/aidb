from dataclasses import dataclass
from enum import Enum
from typing import Any, Union, List

import sqlalchemy
import sqlglot.expressions as exp

class ExpressionType(Enum):
  COLUMN = "column"
  LITERAL = "literal"
  SUBQUERY = "subquery"

@dataclass
class Expression:
  type: ExpressionType  # column, literal, or select
  value: Any  # if column then, table.column form

@dataclass
class FilteringClause:
  is_negation: bool
  op: exp
  left_exp: Expression
  # in case of boolean column, right expression can be none
  right_exp: Union[Expression, List[Expression], None]

@dataclass
class FilteringPredicate:
  is_negation: bool
  predicate: exp


def change_literal_type_to_col_type(t, v):
  if t == sqlalchemy.INTEGER:
    return int(v)
  elif t == sqlalchemy.FLOAT:
    return float(v)
  else:
    return v


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
  elif node == exp.In:
    return "IN"
  else:
    raise NotImplementedError


def predicate_to_str(p: FilteringClause):
  def exp_to_str(e: Union[Expression, List[Expression]]):
    if isinstance(e, list):
      return "(" + ", ".join([str(exp_to_str(x)) for x in e]) + ")"
    elif e.type == ExpressionType.COLUMN:
      return e.value
    elif e.type == ExpressionType.LITERAL:
      try:
        val = float(e.value)
        if '.' in e.value:
          return val
        else:
          return int(val)
      except ValueError:
        return f"'{e.value}'"
    elif e.type == ExpressionType.SUBQUERY:
      return f"({e.value.sql_query_text})"
    else:
      raise NotImplementedError

  if p.right_exp is None:
    sql_expr = str(p.left_exp.value)
  else:
    sql_expr = f"{exp_to_str(p.left_exp)} {get_expr_string(p.op)} {exp_to_str(p.right_exp)}"
  if p.is_negation:
    sql_expr = "NOT " + sql_expr
  return sql_expr