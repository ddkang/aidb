from aidb.engine.full_scan_engine import FullScanEngine
from aidb.utils.asyncio import asyncio_run
from aidb.query.query import Query

class Engine(FullScanEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    parsed_query = Query(query, self._config)
    res = asyncio_run(self.execute_full_scan(parsed_query, **kwargs))
    return res