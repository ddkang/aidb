from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Dict, List, NamedTuple, Tuple

import networkx as nx
import sqlalchemy


# TODO: Unclear if there's a way to do this with sqlalchemy types
class ColumnType(Enum):
  Integer = 'int'
  Float = 'float'
  String = 'str'
  Boolean = 'bool'

  def parse(col_type):
    col_type = str(col_type).lower()
    if col_type in ['int', 'integer']:
      return ColumnType.Integer
    elif col_type in ['float', 'double']:
      return ColumnType.Float
    elif col_type in ['str', 'string', 'varchar', 'text']:
      return ColumnType.String
    elif col_type in ['bool', 'boolean']:
      return ColumnType.Boolean
    else:
      raise ValueError(f'Unknown column type {col_type}')


def python_type_to_sqlalchemy_type(python_type):
  if python_type == int:
    return sqlalchemy.Integer
  elif python_type == float:
    return sqlalchemy.Float
  elif python_type == str:
    return sqlalchemy.String
  elif python_type == bool:
    return sqlalchemy.Boolean
  else:
    raise ValueError(f'Unknown python type {python_type}')


Column = sqlalchemy.schema.Column
Graph = nx.DiGraph


class InferenceBinding(NamedTuple):
  input_columns: Tuple[str]
  output_columns: Tuple[str]


# TODO: think about this architecture
@dataclass
class Table:
  _table: sqlalchemy.Table

  @cached_property
  def name(self) -> str:
    return self._table.name.lower()

  @cached_property
  def primary_key(self) -> List[str]:
    return self._table.primary_key.columns.keys()

  # Name -> Column
  @cached_property
  def columns(self) -> Dict[str, Column]:
    # Note: this is not actually a dict but I think this is fine
    return self._table.columns

  @cached_property
  def foreign_keys(self) -> Dict[str, str]:
    fkeys = {}
    for col in self._table.columns:
      for fk in col.foreign_keys:
        fkeys[col.name] = fk.target_fullname
    return fkeys
