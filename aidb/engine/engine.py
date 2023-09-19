from aidb.engine.full_scan_engine import FullScanEngine
from aidb.utils.asyncio import asyncio_run


class Engine(FullScanEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    res = asyncio_run(self.execute_full_scan(query, **kwargs))
    return res