from sqlalchemy.sql import text

from aidb.engine.base_engine import BaseEngine
from aidb.query.query import Query


class NonSelectQueryEngine(BaseEngine):
  async def execute_non_select(self, query: Query):
    '''
    Executes non select query
    '''
    # The query is irrelevant since we do a full scan anyway
    async with self._sql_engine.begin() as conn:
      res = await conn.execute(text(query.sql_query_text))
      return res
