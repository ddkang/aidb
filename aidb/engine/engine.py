from typing import List

from aidb.engine.aggregate_engine import AggregateEngine
from aidb.engine.full_scan_engine import FullScanEngine
from aidb.query.query import Query
from aidb.utils.asyncio import asyncio_run


class Engine(FullScanEngine, AggregateEngine):

  def get_tables(self, columns: List[str]) -> List[str]:
    '''Fill tables'''
    tables = set()
    for col in columns:
      table_name = col.split('.')[0]
      tables.add(table_name)
    return list(tables)

  def get_join_str(self, tables: List[str]) -> str:
      table_pairs = [(tables[i], tables[i+1]) for i in range(len(tables) - 1)]
      join_strs = []
      for table1_name, table2_name in table_pairs:
        table1 = self._config.tables[table1_name]
        join_cols = []
        for fkey in table1.foreign_keys:
          if fkey.startswith(table2_name + '.'):
            join_cols.append((fkey, fkey.replace(table2_name, table1_name, 1)))
        join_strs.append(f'INNER JOIN {table2_name} ON {" AND ".join([f"{col1} = {col2}" for col1, col2 in join_cols])}')
      return '\n'.join(join_strs)

  def execute(self, query_str: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    query = Query(query_str)
    if query.is_approx_agg_query():
      return asyncio_run(self.execute_aggregate_query(query))
    else:
      return asyncio_run(self.execute_full_scan(query_str, **kwargs))