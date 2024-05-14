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

  async def clear_ml_cache(self, service_name_list = None):
    '''
    Clear the cache and output table if the ML model has changed.
    Delete the tables following the inference services' topological order to maintain integrity during deletion.
    service_name_list: the name of all the changed services. A list of str or None.
    If the service name list is not given, the output for all the services will be cleared.
    '''
    async with self._sql_engine.begin() as conn:
      service_ordering = self._config.inference_topological_order
      if service_name_list is None:
        service_name_list = [bounded_service.service.name for bounded_service in service_ordering]
      service_name_list = set(service_name_list)
      for bounded_service in reversed(service_ordering):
        if isinstance(bounded_service, CachedBoundInferenceService):
          if bounded_service.service.name in service_name_list:
            for input_column in bounded_service.binding.input_columns:
              service_name_list.add(input_column.split('.')[0])
            asyncio_run(conn.execute(delete(bounded_service._cache_table)))
            for output_column in bounded_service.binding.output_columns:
              asyncio_run(conn.execute(delete(bounded_service._tables[output_column.split('.')[0]]._table)))
        else:
          logger.debug(f"Service binding for {bounded_service.service.name} is not cached")