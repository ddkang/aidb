from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, List, Tuple

import networkx as nx
import sqlalchemy

from aidb.config.config_types import Column, Graph, InferenceBinding, Table
from aidb.inference.bound_inference_service import (
    BoundInferenceService, CachedBoundInferenceService)
from aidb.inference.inference_service import InferenceService
from aidb.utils.constants import (BLOB_TABLE_NAMES_TABLE, CACHE_PREFIX,
                                  CONFIG_PREFIX)
from aidb.utils.logger import logger


@dataclass
class Config:
  '''
  Data class that holds all the information required for an AIDB instance.
  Although the data class is mutable, none of the fields should be mutated
  externally.

  The cached_properties must be deleted when Config is mutated.

  '''
  # Inference services. Required
  inference_services: Dict[str, InferenceService] = field(default_factory=dict)
  inference_bindings: List[BoundInferenceService] = field(default_factory=list)

  # Metadata
  db_uri: str = ''
  blob_tables: List[str] = field(default_factory=list)
  # table name -> blob key (possibly composite)
  blob_keys: Dict[str, List[str]] = field(default_factory=dict)

  # Schema
  tables: Dict[str, Table] = field(default_factory=dict)
  columns: Dict[str, Column] = field(default_factory=dict)
  relations: Dict[str, str] = field(default_factory=dict) # left -> right



  @cached_property
  def inference_graph(self) -> Graph:
    '''
    The inference graph _nodes_ are columns. The _edges_ are inference services.
    '''
    graph = nx.DiGraph()
    for column_name in self.columns.keys():
      graph.add_node(column_name)

    for bound_service in self.inference_bindings:
      binding = bound_service.binding
      for inp in binding.input_columns:
        for out in binding.output_columns:
          graph.add_edge(inp, out, bound_service=bound_service)
    return graph


  @cached_property
  def table_graph(self) -> Graph:
    '''
    Directed graph of foreign key relationship between tables.
    The table graph _nodes_ are tables. The _edges_ are foreign key relations.
    If A -> B, then A has a foreign key that refers to B's primary key.
    '''
    table_graph = nx.DiGraph()
    for table_name in self.tables:
      for fk_col, pk_other_table in self.tables[table_name].foreign_keys.items():
        parent_table = pk_other_table.split('.')[0]
        table_graph.add_edge(table_name, parent_table)
    return table_graph


  @cached_property
  def inference_topological_order(self) -> List[BoundInferenceService]:
    '''
    Returns a topological ordering of the inference services.  Note that this is
    not necessarily the same as the order in which AIDB runs the inference
    services depending on the query.
    '''
    graph = self.inference_graph
    column_order = nx.topological_sort(graph)
    binding_order = []
    seen_binding_idxes = set()
    for node in column_order:
      edges = graph.in_edges(node)
      for edge in edges:
        properties = graph.get_edge_data(*edge)
        bound_service = properties['bound_service']
        if bound_service in seen_binding_idxes:
          continue
        binding_order.append(bound_service)
        seen_binding_idxes.add(bound_service)

    return binding_order


  @cached_property
  def dialect(self):
    # TODO: fix this, copied from base_engine
    dialect = self.db_uri.split(':')[0]
    if '+' in dialect:
      dialect = dialect.split('+')[0]
    return dialect


  @cached_property
  def column_by_service(self) -> Dict[str, Tuple[InferenceBinding, InferenceService]]:
    raise NotImplementedError()


  @cached_property
  def relations_by_table(self) -> Dict[str, List[str]]:
    raise NotImplementedError()


  def _check_blob_table(self):
    '''
    Check if the blob table is valid. It must satisfy the following conditions:
    1. There must be at least one blob table.
    2. The blob table must exist in the database schema.
    3. The blob table's primary key must match the blob keys in metadata.
    '''
    if len(self.blob_tables) == 0:
      raise Exception('No blob table defined')

    for blob_table in self.blob_tables:
      if blob_table not in self.tables:
        raise Exception(f'{blob_table} doesn\'t exist in database schema')

      metadata_blob_key_set = set(self.blob_keys[blob_table])
      primary_key_set = set([f'{blob_table}.{k}' for k in self.tables[blob_table].primary_key])
      if metadata_blob_key_set != primary_key_set:
        raise Exception(
          f'The actual primary key of {blob_table} doesn\'t match the blob keys in metadata.\n'
          f'Keys present in metadata but missing in primary key: {metadata_blob_key_set - primary_key_set}.\n'
          f'Keys present in primary key but missing in metadata: {primary_key_set - metadata_blob_key_set}.'
        )


  def check_schema_validity(self):
    '''
    Check config schema, including checking blob table and checking if the table relations form a DAG.
    '''
    self._check_blob_table()

    if not nx.is_directed_acyclic_graph(self.table_graph):
      raise Exception('Invalid Table Schema: Table relations can not have cycle')

    # TODO: check inference service validity


  def clear_cached_properties(self):
    # Need the keys because the cached properties are only created when they are accessed.
    keys = [key for key, value in vars(Config).items() if isinstance(value, cached_property)]
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
        table_cols[str(column)] = column
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
    logger.info(f'Adding inference service {service_name}')
    self.inference_services[service_name] = service


  def bind_inference_service(self, bound_service: BoundInferenceService):
    '''
    Both the inputs and outputs are lists of column names. The ordering is critical.

    The cached properties are cleared, so the toplogical sort and columns by service are updated.
    '''
    self.clear_cached_properties()
    self.inference_bindings.append(bound_service)
