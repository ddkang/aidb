from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from aidb.config.config_types import Column, Table


@dataclass
class Config:
  '''
  Data class that holds all the information required for an AIDB instance.
  Although the data class is mutable, none of the fields should be mutated externally.
  '''

  # Metadata
  db_uri: str = ''
  blob_tables: List[str] = field(default_factory=list)
  # table name -> blob key (possibly composite)
  blob_keys: Dict[str, List[str]] = field(default_factory=dict)

  # Schema
  tables: Dict[str, Table] = field(default_factory=dict)
  columns: Dict[str, Column] = field(default_factory=dict)
  relations: Dict[str, str] = field(default_factory=dict) # left -> right

  # Inference engines
  # TODO: inference engine type
  engine_by_name: Dict[str, str] = field(default_factory=dict)
  # engine name -> (inputs, outputs)
  engine_bindings: Dict[str, Tuple[List[str], List[str]]] = field(default_factory=dict)
  # TODO: inference engine type
  column_by_engine: Dict[str, str] = field(default_factory=dict)

  # Derived
  table_graph: Dict[str, str] = field(default_factory=dict)


  # TODO: actually check validity
  def check_validity(self):
    return True