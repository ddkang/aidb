from aidb.engine.full_scan_engine import FullScanEngine


class Engine(FullScanEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    return super().execute(query, **kwargs)