from dataclasses import dataclass
from typing import Optional


@dataclass
class ColumnInfo:
  name: str
  is_primary_key: bool
  refers_to: Optional[tuple]  # (table, column)
  d_type = None


def extract_column_info(table_name, column_str) -> ColumnInfo:
  pk = False
  if column_str.startswith("pk_"):
    pk = True
    column_str = column_str[3:]  # get rid of pk_ prefix
  t, c = column_str.split('.')
  fk = None
  if t != table_name:
    fk = (t, c)
  return ColumnInfo(c, pk, fk)
