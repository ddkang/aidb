from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


def _get_normalized_column_name(table: str, column: str) -> str:
  return f'{table}.{column}'.lower()


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


@dataclass
class Column:
  table: str
  name: str
  type: ColumnType
  is_primary_key: bool

  @property
  def full_name(self) -> str:
    return _get_normalized_column_name(self.table, self.name)


# TODO: think about this architecture
@dataclass
class Table:
  name: str
  columns: Dict[str, Column] # name -> type
  primary_key: List[str]
  foreign_keys: Dict[str, str] # name -> table.col