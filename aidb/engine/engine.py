from aidb.engine.limit_engine import LimitEngine
from aidb.utils.asyncio import asyncio_run
from aidb.query.query import Query


class Engine(LimitEngine):
  def execute(self, query: str, **kwargs):
    '''
    Executes a query and returns the results.
    '''
    # TODO: branch based on query type
    #query = '''SELECT objects00.x_max FROM objects00 GROUP BY objects00.frame HAVING COUNT(object_name = 'car') > 1 AND COUNT(object_name = 'car') < 3 OR COUNT(object_name = 'car') = 3;'''
    #query = '''SELECT objects00.frame FROM colors02 join objects00 on colors02.frame = objects00.frame AND colors02.object_id = objects00.object_id GROUP BY objects00.frame, objects00.object_id HAVING COUNT(objects00.object_name = 'car') > 1;'''
    #query = '''SELECT colors02.frame AS test, lights01.light_4 AS lll FROM colors02 join counts03 on colors02.frame = counts03.frame join  lights01 on colors02.frame = lights01.frame  WHERE lll = 'red' AND colors02.color='red' OR 0 < counts03.count < 10 OR counts03.frame > 100;'''
    # query = '''SELECT * FROM colors02 WHERE colors02.color = 'red' AND colors02.frame IN (SELECT objects00.frame FROM objects00);'''
    parsed_query = Query(query, self._config)
    #print('1', parsed_query.filtering_predicates)
    # fp = parsed_query.having_predicates
    # print('2', fp)
    # print('engine', parsed_query.inference_engines_required_for_filtering_predicates(fp))
    # print(parsed_query.having_predicates)
    if parsed_query.is_limit_query():
      return asyncio_run(self._execute_limit_query(parsed_query, **kwargs))
    else:
      return asyncio_run(self.execute_full_scan(parsed_query, **kwargs))
