import sqlalchemy
from dataclasses import dataclass
from typing import Any, Union
import sqlglot.expressions as exp


def change_literal_type_to_col_type(t, v):
  if t == sqlalchemy.INTEGER:
    return int(v)
  elif t == sqlalchemy.FLOAT:
    return float(v)
  else:
    return v

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
