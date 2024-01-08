from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Dict, List, NamedTuple, Tuple

import networkx as nx
import sqlalchemy


AIDBListType = type('AIDBListType', (), {})


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
  copy_map: Dict[str, str] = {}

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


def python_type_to_sqlalchemy_type(python_type):
  if python_type == int:
    return sqlalchemy.Integer
  elif python_type == float:
    return sqlalchemy.Float
  # TODO: think if this is the best way.
  elif python_type == str or python_type == object:
    return sqlalchemy.String
  elif python_type == bool:
    return sqlalchemy.Boolean
  else:
    raise ValueError(f'Unknown python type {python_type}')
