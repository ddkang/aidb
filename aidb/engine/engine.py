from aidb.engine.approx_aggregate_engine import ApproximateAggregateEngine
from aidb.engine.limit_engine import LimitEngine
from aidb.engine.non_select_query_engine import NonSelectQueryEngine
from aidb.utils.asyncio import asyncio_run
from aidb.query.query import Query


class Engine(LimitEngine, ApproximateAggregateEngine, NonSelectQueryEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    try:
      parsed_query = Query(query, self._config)
      if parsed_query.is_approx_agg_query:
        return asyncio_run(self.execute_aggregate_query(parsed_query, **kwargs))
      elif parsed_query.is_limit_query():
        return asyncio_run(self._execute_limit_query(parsed_query, **kwargs))
      elif parsed_query.is_select_query():
        return asyncio_run(self.execute_full_scan(parsed_query, **kwargs))
      else:
        return asyncio_run(self.execute_non_select(parsed_query))
    except Exception as e:
      raise e
