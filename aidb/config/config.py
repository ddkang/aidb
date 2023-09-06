from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List

import networkx as nx
import sqlalchemy

from aidb.config.config_types import Column, Graph, InferenceBinding, Table
from aidb.inference.inference_service import InferenceService
from aidb.utils.constants import (BLOB_TABLE_NAMES_TABLE, CACHE_PREFIX,
                                  CONFIG_PREFIX)


@dataclass
class Config:
  '''
  Data class that holds all the information required for an AIDB instance.
  Although the data class is mutable, none of the fields should be mutated
  externally.

  The cached_properties must be deleted when Config is mutated.

  '''
  # Inference services. Required
  inference_services: Graph

  # Metadata
  db_uri: str = ''
  blob_tables: List[str] = field(default_factory=list)
  # table name -> blob key (possibly composite)
  blob_keys: Dict[str, List[str]] = field(default_factory=dict)

  # Schema
  tables: Dict[str, Table] = field(default_factory=dict)
  columns: Dict[str, Column] = field(default_factory=dict)
  relations: Dict[str, str] = field(default_factory=dict) # left -> right

  # inference service name -> List[(inputs, outputs)]
  inference_bindings: Dict[str, List[InferenceBinding]] = field(default_factory=dict)


  # TODO: figure out the type
  @cached_property
  def table_graph(self) -> Dict[str, str]:
    raise NotImplementedError()


  @cached_property
  def inference_topological_order(self) -> List[str]:
    '''
    Returns a topological ordering of the inference services.  Note that this is
    not necessarily the same as the order in which AIDB runs the inference
    services depending on the query.
    '''
    topo = nx.topological_sort(self.inference_services)
    raise NotImplementedError()


  @cached_property
  def column_by_service(self) -> Dict[str, InferenceService]:
    raise NotImplementedError()


  # TODO: actually check validity
  def check_validity(self):
    return True


  def clear_cached_properties(self):
    # TODO: is there some way to automatically find the cached properties?
    # Need the keys because the cached properties are only created when they are accessed.
    keys = [
      'table_graph',
      'inference_topological_order',
      'column_by_service',
    ]
    for key in keys:
      if key in self.__dict__:
        del self.__dict__[key]


  # Mutators. The cached properties must be cleared after these are called.
  def load_from_sqlalchemy(self, conn: sqlalchemy.engine.base.Connection):
    '''
    Loads the tables, columns, and relations from a sqlalchemy connection.
    '''
    self.clear_cached_properties()

    metadata = sqlalchemy.MetaData()
    # TODO: should base go into config?
    Base = sqlalchemy.ext.automap.automap_base()
    Base.prepare(conn, reflect=True)

    aidb_tables: Dict[str, Table] = {}
    aidb_cols = {}
    aidb_relations = {}

    for table_name in Base.classes.keys():
      if table_name.startswith(CONFIG_PREFIX):
        continue
      if table_name.startswith(CACHE_PREFIX):
        continue

      sqlalchemy_table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=conn)
      table_cols = {}
      for column in sqlalchemy_table.columns:
        aidb_cols[str(column)] = column
        table_cols[column.name] = column
        for fk in column.foreign_keys:
          aidb_relations[str(fk.column)] = str(column)
          aidb_relations[str(column)] = str(fk.column)

      aidb_tables[table_name] = Table(
        sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=conn),
      )

    blob_metadata_table = sqlalchemy.Table(BLOB_TABLE_NAMES_TABLE, metadata, autoload=True, autoload_with=conn)
    try:
      blob_keys_flat = conn.execute(blob_metadata_table.select()).fetchall()
      blob_tables = list(set([row['table_name'] for row in blob_keys_flat]))
      blob_tables.sort()
      blob_keys = defaultdict(list)
      for row in blob_keys_flat:
        full_name = f'{row["table_name"]}.{row["blob_key"]}'
        blob_keys[row['table_name']].append(full_name)
      for table in blob_keys:
        blob_keys[table].sort()
    except:
      raise ValueError(f'Could not find blob metadata table {BLOB_TABLE_NAMES_TABLE} or table is malformed')

    # TODO: load metadata columns
    # TODO: check the engines
    # TODO: check that the cache tables are valid

    self.tables = aidb_tables
    self.columns = aidb_cols
    self.relations = aidb_relations
    self.blob_tables = blob_tables
    self.blob_keys = blob_keys


  def add_inference_service(self, service_name: str, service: InferenceService):
    self.clear_cached_properties()
    self.inference_services.add_node(service_name, service=service)


  def bind_inference_service(self, service_name: str, inputs: List[str], outputs: List[str]):
    '''
    Both the inputs and outputs are lists of column names. The ordering is critical.

    The cached properties are cleared, so the toplogical sort and columns by service are updated.
    '''
    self.clear_cached_properties()
    self.inference_bindings[service_name].append(InferenceBinding(inputs, outputs))