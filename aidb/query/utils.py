import sqlalchemy
from dataclasses import dataclass
from typing import Any, Union
import sqlglot.expressions as exp


@dataclass
class Expression:
  type: str  # column or literal
  value: Any  # if column then, table.column form


@dataclass
class FilteringClause:
  is_negation: bool
  op: exp
  left_exp: Expression
  # in case of boolean column, right expression can be none
  right_exp: Union[Expression, None]


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
  else:
    raise NotImplementedError


def predicate_to_str(p: FilteringClause):
  def exp_to_str(e: Expression):
    if e.type == "column":
      return e.value
    elif e.type == "literal":
      try:
        val = float(e.value)
        if '.' in e.value:
          return val
        else:
          return int(val)
      except ValueError:
        return f"'{e.value}'"
    else:
      raise NotImplementedError

  if p.right_exp is None:
    sql_expr = str(p.left_exp.value)
  else:
    sql_expr = f"{exp_to_str(p.left_exp)} {get_expr_string(p.op)} {exp_to_str(p.right_exp)}"
  if p.is_negation:
    sql_expr = "NOT " + sql_expr
  return sql_expr
