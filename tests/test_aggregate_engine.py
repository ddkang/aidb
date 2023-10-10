from multiprocessing import Process
import os
import unittest
from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, MAX_DIFF_AIDB_GT_RESULT


DB_URL = "sqlite+aiosqlite://"
queries = [
      (
        'aggregate',
        '''SELECT AVG(x_min) FROM objects00 ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT AVG(x_min) FROM objects00;'''
       ),
      (
        'aggregate',
        '''SELECT SUM(x_min) FROM objects00 ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT SUM(x_min) FROM objects00;'''
      ),
      (
        'aggregate',
        '''SELECT COUNT(x_min) FROM objects00 ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT COUNT(x_min) FROM objects00;'''
      ),
      (
        'aggregate',
        '''SELECT COUNT(*) FROM objects00 ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT COUNT(*) FROM objects00;'''
      ),
      (
         'aggregate',
         '''SELECT AVG(x_min) FROM objects00 
            WHERE frame < 10000 
            ERROR_TARGET 0.01 CONFIDENCE 95;''', 
         '''SELECT AVG(x_min) FROM objects00 
            WHERE frame < 10000;''' 
      ),
      (
        'aggregate',
        '''SELECT AVG(x_min) FROM objects00 
            WHERE object_name='car' AND frame < 10000
            ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT AVG(x_min) FROM objects00 
            WHERE object_name='car' AND frame < 10000;'''
      ),
     (
       'aggregate',
       '''SELECT AVG(x_min) FROM objects00 
          WHERE object_id > 0 
          ERROR_TARGET 0.01 CONFIDENCE 95;''', 
       '''SELECT AVG(x_min) FROM objects00 
          WHERE object_id > 0;''' 
     ),
     (
        'aggregate',
        '''SELECT COUNT(frame) FROM counts03 
            WHERE frame >= 1000
            ERROR_TARGET 0.01 CONFIDENCE 95;''',
        '''SELECT COUNT(frame) FROM counts03
            WHERE frame >= 1000;''',
      ),
     (
        'aggregate',
        '''SELECT COUNT(frame) FROM lights01 
           WHERE light_1 = 'green' ERROR_TARGET 0.01 CONFIDENCE 95;''', 
        '''SELECT COUNT(frame) FROM lights01 
           WHERE light_1 = 'green';''' 
      )
    ]

class AggeregateEngineTests(IsolatedAsyncioTestCase):
  def _equality_check(self, aidb_res, gt_res):
    assert len(aidb_res) == len(gt_res)
    aidb_res, gt_res = aidb_res[0][0], gt_res[0][0]
    if abs(aidb_res - gt_res) / (gt_res) < MAX_DIFF_AIDB_GT_RESULT:
      return True
    return False

  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()

    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
    register_inference_services(aidb_engine, data_dir)
    for query_type, aidb_query, aggregate_query in queries:
      print(f'Running query {aggregate_query} in ground truth database')
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(aggregate_query))
        gt_res = gt_res.fetchall()

      print(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      print(f'aidb_res: {aidb_res}, gt_res: {gt_res}')
      assert self._equality_check(aidb_res, gt_res)

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()