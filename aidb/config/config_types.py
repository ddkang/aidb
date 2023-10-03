from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Dict, List, NamedTuple, Tuple, Optional

import networkx as nx
import sqlalchemy
import pandas as pd
import numpy as np


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


class VectorDatabaseType(Enum):
  FAISS = 'FAISS'
  CHROMA = 'Chroma'
  WEAVIATE = 'Weaviate'


@dataclass
class WeaviateAuth:
  """
  :param url: weaviate url
  :param username: weaviate username
  :param pwd: weaviate password
  :param api_key: weaviate api key, user should choose input either username/pwd or api_key
  """
  url: Optional[str] = field(default=None)
  username: Optional[str] = field(default=None)
  pwd: Optional[str] = field(default=None)
  api_key: Optional[str] = field(default=None)


@dataclass
class TastiConfig:
  '''
  :param index_name: vector database index name
  :param blob_ids: blob index in blob table, it should be unique for each data record
  :param nb_buckets: number of buckets for FPF, it should be same as the number of buckets for oracle
  :param vector_database: vector database type, it should be FAISS, Chroma or Weaviate
  :param weaviate_auth: Weaviate authentification
  :param index_path: vector database(FAISS, Chroma) index path, path to store database
  :param percent_fpf: percent of randomly selected buckets in FPF
  :param seed: random seed
  '''
  index_name: str
  blob_ids: pd.DataFrame
  nb_buckets: int
  vector_database_name: VectorDatabaseType = field(default=VectorDatabaseType.FAISS)
  percent_fpf: float = 0.75
  seed: int = 1234
  weaviate_auth: Optional[WeaviateAuth] = field(default=None)
  index_path: Optional[str] = field(default=None)
  reps: Optional[np.ndarray] = field(default=None)