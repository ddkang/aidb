from aidb.engine.full_scan_engine import FullScanEngine


class Engine(FullScanEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    res = self._loop.run_until_complete(self.execute_full_scan(query, **kwargs))
    return res