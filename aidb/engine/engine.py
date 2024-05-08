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

  async def clear_ml_cache(self):
    '''
    Clear the cache and output table if the ML model has changed.
    For each cached inference service, build the reference graph of the tables based on fk constraints,
    and then delete the tables following the graph's topological order to maintain integrity during deletion.
    '''
    for inference_binding in self._config.inference_bindings:
      if isinstance(inference_binding, CachedBoundInferenceService):
        async with inference_binding._engine.begin() as conn:
          tables_to_delete = inference_binding.get_tables(inference_binding.binding.output_columns) + [inference_binding._cache_table_name]
          fk_ref_graph = nx.DiGraph()
          fk_ref_graph.add_nodes_from(tables_to_delete)
          for table_name in tables_to_delete:
            if table_name == inference_binding._cache_table_name:
              table = inference_binding._cache_table
            else:
              table = inference_binding._tables[table_name]._table
            for constraint in table.constraints:
              if isinstance(constraint, ForeignKeyConstraint):
                for fk in constraint.elements:
                  fk_ref_table_name = fk.column.table.name
                  if fk_ref_table_name in fk_ref_graph:
                    fk_ref_graph.add_edge(table_name,fk_ref_table_name)
          
          table_order = nx.topological_sort(fk_ref_graph)
          for table_name in table_order:
            if table_name == inference_binding._cache_table_name:
              table = inference_binding._cache_table
            else:
              table = inference_binding._tables[table_name]._table
            await conn.execute(delete(table))
          
      else:
        logger.debug(f"Service binding for {inference_binding.service.name} is not cached")