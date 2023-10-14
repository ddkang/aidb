from aidb.engine.aggregate_engine import ApproximateAggregateEngine
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.utils.asyncio import asyncio_run
from aidb.query.query import Query

class Engine(FullScanEngine, ApproximateAggregateEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    parsed_query = Query(query, self._config)
    
    if parsed_query.is_approx_agg_query:
    	return asyncio_run(self.execute_aggregate_query(parsed_query, **kwargs))

    return asyncio_run(self.execute_full_scan(parsed_query, **kwargs))