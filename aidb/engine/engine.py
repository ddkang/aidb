from aidb.engine.aggregate_engine import AggregateEngine
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.utils.asyncio import asyncio_run


class Engine(FullScanEngine, AggregateEngine):
  def execute(self, query_str: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    if True: # query.is_approx_agg_query():
    	return asyncio_run(self.execute_aggregate_query(query_str))
    else:
	    return asyncio_run(self.execute_full_scan(query_str, **kwargs))