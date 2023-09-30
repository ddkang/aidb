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
  relations: Dict[str, str] = field(default_factory=dict)  # left -> right



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
  def column_graph(self) -> Graph:
    '''
    Directed graph of foreign key relationship between columns.
    The column graph _nodes_ are columns. The _edges_ are foreign key relations.
    If A -> B, then A is a foreign key column that refers to B.
    '''
    column_graph = nx.DiGraph()
    for table_name in self.tables:
      for fk_col, pk_other_table in self.tables[table_name].foreign_keys.items():
        column_graph.add_edge(f"{table_name}.{fk_col}", pk_other_table)
    return column_graph


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
  def column_by_service(self) -> Dict[str, BoundInferenceService]:
    '''
    Returns a dictionary mapping output column names to the inference service that produces them.
    '''
    column_service = dict()
    for bound_service in self.inference_bindings:
      for output_col in bound_service.binding.output_columns:
        if output_col in column_service:
          raise Exception(f'Column {output_col} is bound to multiple services')
        else:
          column_service[output_col] = bound_service
    return column_service


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


  def _check_foreign_key_refers_to_primary_key(self, input_tables, output_tables):
    '''
    1. Each output table should include the minimal set of primary key columns from the input tables.
    2. To ensure that no primary key column in the output table is null, any column in the output table
       with a foreign key relationship must exist in the primary key columns of the input tables.
    '''
    # Assumption:
    # 1. If table A can join table B, then the join keys are those columns that have same name in both table A and B.
    # 2. If table C is a derived table of table A, then the foreign key columns of table C have same name as
    # corresponding columns in table A. e.x. objects.frame -> blob.frame
    input_primary_key_columns = set()
    for input_table in input_tables:
      for pk_col in self.tables[input_table].primary_key:
        input_primary_key_columns.add(f"{input_table}.{pk_col}")

    for output_table in output_tables:
      out_foreign_key_columns = set()
      out_primary_key_columns = set()

      # Each output table should include the minimal set of primary key columns from the input tables.
      for fk_col, refers_to in self.tables[output_table].foreign_keys.items():
        if fk_col in self.tables[output_table].primary_key:
          if refers_to not in input_primary_key_columns:
            raise Exception(f'{output_table} primary key column {fk_col} is not in input tables')
          out_foreign_key_columns.add(refers_to)

      for pk_col in self.tables[output_table].primary_key:
        out_primary_key_columns.add(f"{output_table}.{pk_col}")

      # Any column in the output table with a foreign key relationship
      # must exist in the primary key columns of the input tables.
      for pk_col in input_primary_key_columns:
        if pk_col not in out_foreign_key_columns and pk_col not in out_primary_key_columns:
          current_table = pk_col.split('.')[0]
          current_pk_col = pk_col.split('.')[1]
          raise_exception = True
          while current_pk_col in self.tables[current_table].foreign_keys:
            if current_pk_col in self.tables[current_table].foreign_keys:
              current_pk_col = self.tables[current_table].foreign_keys[current_pk_col]
              current_table = current_pk_col.split('.')[0]
            if current_pk_col in out_foreign_key_columns or current_pk_col in out_primary_key_columns:
              raise_exception = False
              break
          if raise_exception:
            raise Exception(f'Primary key column {pk_col} in input table is not refered by output table {output_table}')


  def check_inference_service_validity(self, bound_inference: BoundInferenceService):
    '''
    Check if the inference service is valid whenever adding a bound inference.
    It must satisfy the following conditions:
    1. The inference service must be defined in config.
    2. The input columns and output columns must exist in the database schema.
    3. The output column must be bound to only one inference service.
    4. The input table must include the minimal set of primary key columns from the output table.
       And to ensure that no primary key column in the output table is null, any column in the output table.
       with a foreign key relationship must exist in the primary key columns of the input tables.
    5. The graphs of table relations and column relations must form DAGs.
    '''

    # The inference service must be defined in config.
    if bound_inference.service.name not in self.inference_services:
      raise Exception(f'Inference service {bound_inference.service.name} is not defined in config')

    input_tables = set()
    output_tables = set()
    binding = bound_inference.binding

    # The input columns and output columns must exist in the database schema.
    if not binding.input_columns or not binding.output_columns:
      raise Exception(f'Inference service {bound_inference.service.name} has no input columns or output columns')

    for column in binding.input_columns:
      if column not in self.columns:
        raise Exception(f'Input column {column} doesn\'t exist in database')
      input_tables.add(column.split('.')[0])

    for column in binding.output_columns:
      if column not in self.columns:
        raise Exception(f'Output column {column} doesn\'t exist in database')
      output_table = column.split('.')[0]
      output_tables.add(output_table)

    # Check if the output column is bound to only one inference service
    self.column_by_service

    # The input table must include the minimal set of primary key columns from the output table.
    # And to ensure that no primary key column in the output table is null, any column in the output table.
    # with a foreign key relationship must exist in the primary key columns of the input tables.
    self._check_foreign_key_refers_to_primary_key(input_tables, output_tables)

    # The graphs of table relations and column relations must form DAGs.
    if not nx.is_directed_acyclic_graph(self.table_graph) or not nx.is_directed_acyclic_graph(self.inference_graph):
      raise Exception(f'Inference service {bound_inference.service.name} will result in cycle in relations')


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
    metadata.reflect(conn)

    aidb_tables: Dict[str, Table] = {}
    aidb_cols = {}
    aidb_relations = {}

    for sqlalchemy_table in metadata.sorted_tables:
      table_name = sqlalchemy_table.name
      if table_name.startswith(CONFIG_PREFIX):
        continue
      if table_name.startswith(CACHE_PREFIX):
        continue

      table_cols = {}
      for column in sqlalchemy_table.columns:
        aidb_cols[str(column)] = column
        table_cols[str(column)] = column
        for fk in column.foreign_keys:
          aidb_relations[str(fk.column)] = str(column)
          aidb_relations[str(column)] = str(fk.column)

      aidb_tables[table_name] = Table(sqlalchemy_table)

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

    self.check_schema_validity()


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
    try:
      self.check_inference_service_validity(bound_service)
    except Exception:
      self.inference_bindings.remove(bound_service)
      self.clear_cached_properties()
      raise
