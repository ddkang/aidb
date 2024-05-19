from collections import deque

import networkx as nx
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.sql import delete

from aidb.engine.approx_aggregate_join_engine import \
    ApproximateAggregateJoinEngine
from aidb.engine.approx_select_engine import ApproxSelectEngine
from aidb.engine.limit_engine import LimitEngine
from aidb.engine.non_select_query_engine import NonSelectQueryEngine
from aidb.inference.bound_inference_service import CachedBoundInferenceService
from aidb.query.query import Query
from aidb.utils.asyncio import asyncio_run
from aidb.utils.logger import logger


class Engine(LimitEngine, NonSelectQueryEngine, ApproxSelectEngine, ApproximateAggregateJoinEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    try:
      parsed_query = Query(query, self._config)
      all_queries = parsed_query.all_queries_in_expressions
      # FIXME: We have many validity checks for different queries.
      #     It's better to put them together and check the validity first.
      # check validity of user defined function query

      if parsed_query.is_udf_query:
        parsed_query.check_udf_query_validity()
      result = None

      for parsed_single_query, _ in all_queries:
        if parsed_single_query.is_approx_agg_query:
          if parsed_single_query.is_aqp_join_query:
            result = asyncio_run(self.execute_aggregate_join_query(parsed_single_query, **kwargs))
          else:
            result = asyncio_run(self.execute_aggregate_query(parsed_single_query, **kwargs))
        elif parsed_single_query.is_approx_select_query:
          result = asyncio_run(self.execute_approx_select_query(parsed_single_query, **kwargs))
        elif parsed_single_query.is_limit_query():
          result = asyncio_run(self._execute_limit_query(parsed_single_query, **kwargs))
        elif parsed_single_query.is_select_query():
          result = asyncio_run(self.execute_full_scan(parsed_single_query, **kwargs))
        else:
          result = asyncio_run(self.execute_non_select(parsed_single_query))

      return result
    except Exception as e:
      raise e
    finally:
      self.__del__()

  async def clear_ml_cache(self, services_to_clear = None):
    '''
    Clear the cache and output table if the ML model has changed.
    1. Collect the output tables directly related to the selected services.
    2. Collect the output tables that need to be cleared considering the fk and service constraints.
    3. Delete the cache tables. 
    4. Delete the output tables in the reversed topological order of table_graph.

    services_to_clear: the name of all the changed services. A list of str or None.
    If the service name list is not given, the output for all the services will be cleared.
    Note that the output for some other services may be also cleared because of fk constraints.
    '''
    if services_to_clear is None:
      services_to_clear = [bound_service.service.name for bound_service in self._config.inference_bindings]
    services_to_clear = set(services_to_clear)
    
    # The services that has output columns in the table
    table_related_service = {table_name: set() for table_name in self._config.tables.keys()}
    # The output tables of each service
    output_tables = {service_name: set() for service_name in self._config.inference_services.keys()}
    tables_to_clear = set()

    for bound_service in self._config.inference_bindings:
      if isinstance(bound_service, CachedBoundInferenceService):
        # Construct the table to service map and the output table list
        service_name = bound_service.service.name
        for output_column in bound_service.binding.output_columns:
          output_tables[service_name].add(output_column.split('.')[0])
        for output_table_name in output_tables[service_name]:
          table_related_service[output_table_name].add(service_name)
        # Collect the output tables directly related to service_to_clear
        if service_name in services_to_clear:
          tables_to_clear.update(output_tables[service_name])
      else:
        logger.debug(f"Service binding for {bound_service.service.name} is not cached")
    
    # Collect the output tables that need to be cleared considering the fk and service constraints
    # Do a bfs on the reversed table graph
    table_graph = self._config.table_graph
    table_queue = deque(tables_to_clear)
    
    def add_table_to_queue(table):
      if table not in tables_to_clear:
        tables_to_clear.add(table)
        table_queue.append(table)

    while len(table_queue) > 0:
      table_name = table_queue.popleft()
      # Add tables considering fk constraints
      for in_edge in table_graph.in_edges(table_name):
        add_table_to_queue(in_edge[0])
      # Add tables considering service constraints
      services_to_clear.update(table_related_service[table_name])
      for service_name in table_related_service[table_name]:
        for table_to_add in output_tables[service_name]:
          add_table_to_queue(table_to_add)
  
    async with self._sql_engine.begin() as conn:
      # Delete cache tables
      for bound_service in self._config.inference_bindings:
        if bound_service.service.name in services_to_clear:
          asyncio_run(conn.execute(delete(bound_service._cache_table)))

      # Delete output tables
      table_order = nx.topological_sort(table_graph)
      for table_name in table_order:
        if table_name in tables_to_clear:
          asyncio_run(conn.execute(delete(self._config.tables[table_name]._table)))