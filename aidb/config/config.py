from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple

from aidb.config.config_types import Column, Table
from aidb.inference.inference_service import InferenceService
from aidb.utils.graph import Graph

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

  # Inference engines
  engine_by_name: Dict[str, InferenceService] = field(default_factory=dict)
  # engine name -> (inputs, outputs)
  engine_bindings: Dict[str, Tuple[List[str], List[str]]] = field(default_factory=dict)
  column_by_engine: Dict[str, InferenceService] = field(default_factory=dict)

  @cached_property
  def relations(self) -> Dict[str, str]:
    relations = {}
    for col in self.columns:
      for fk in self.columns[col].foreign_keys:
        relations[col] = fk.target_fullname
    return relations

  # TODO: figure out the type
  @cached_property
  def table_graph(self) -> Dict[str, List[str]]:
    table_graph = {}
    for table_name in self.tables:
      parent_table_set = set()
      for fk_col, pk_other_table in self.tables[table_name].foreign_keys.items():
        parent_table = pk_other_table.split('.')[0]
        parent_table_set.add(parent_table)
      table_graph[table_name] = list(parent_table_set)
    return table_graph

    # raise NotImplementedError()

  @cached_property
  def engine_graph(self) -> Dict[str, str]:
    raise NotImplementedError()


  def _check_blob_table(self):
    if len(self.blob_tables) == 0:
      raise Exception(
        'No blob table defined'
      )

    for blob_table in self.blob_tables:
      if len(self.table_graph[blob_table]) != 0:
        raise Exception(
          f'{blob_table} shouldn\'t have parent table'
        )

      if blob_table not in self.tables:
        raise Exception(
          f'{blob_table} doesn\'t exist in database schema'
        )

      for primary_key in self.tables[blob_table].primary_key:
        if primary_key not in self.blob_keys[blob_table]:
          raise Exception(
            f'All primary keys of blob table must exist in blob keys'
          )

  def _check_foreign_key_refers_to_primary_key(self):
    for table_name in self.table_graph:
      foreign_key_columns = set()
      for fk_col, pk_other_table in self.tables[table_name].foreign_keys.items():
        foreign_key_columns.add(pk_other_table)
      for parent_table_name in self.table_graph[table_name]:
        for pk_col in self.tables[parent_table_name].primary_key:
          normalized_column_name = f'{parent_table_name}.{pk_col}'
          if normalized_column_name not in foreign_key_columns:
            raise Exception(
              f'{table_name} foreign key relation doesn\'t refer to all primary key columns in {parent_table_name}'
            )


  # TODO: actually check validity
  def check_validity(self):

    self._check_blob_table()
    self._check_foreign_key_refers_to_primary_key()

    g = Graph(self.table_graph)
    if g.isCyclic():
      raise Exception('Invalid Table Schema: Table relations can not have cycle')


