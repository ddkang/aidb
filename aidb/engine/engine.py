from aidb.engine.approx_aggregate_engine import ApproximateAggregateEngine
from aidb.engine.approx_select_engine import ApproxSelectEngine
from aidb.engine.limit_engine import LimitEngine
from aidb.engine.non_select_query_engine import NonSelectQueryEngine
from aidb.utils.asyncio import asyncio_run
from aidb.query.query import Query


class Engine(LimitEngine, ApproximateAggregateEngine, NonSelectQueryEngine, ApproxSelectEngine):
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
      _ = parsed_query.is_udf_query_valid
      result = None

      for parsed_single_query, _ in all_queries:
        if parsed_single_query.is_approx_agg_query:
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
