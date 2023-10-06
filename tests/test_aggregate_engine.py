from multiprocessing import Process
import os
import unittest
from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine


DB_URL = "sqlite+aiosqlite://"
queries = [
      (
        'aggregate',
        '''SELECT AVG(x_min) FROM objects00;''',
        '''SELECT AVG(x_min) FROM objects00;'''
       ),
      (
        'aggregate',
        '''SELECT SUM(x_min) FROM objects00;''',
        '''SELECT SUM(x_min) FROM objects00;'''
      ),
      (
        'aggregate',
        '''SELECT COUNT(x_min) FROM objects00;''',
        '''SELECT COUNT(x_min) FROM objects00;'''
      )
      # ,
     # (
     #   'aggregate',
     #   '''SELECT AVG(x_min) FROM objects00 WHERE object_id > 0;''', 
     #   '''SELECT AVG(x_min) FROM objects00 WHERE object_id > 0;''' 
     # ),
     # (
      #   'aggregate',
      #   '''SELECT COUNT(frame) FROM lights01 WHERE light_1 = 'green';''', 
      #   '''SELECT COUNT(frame) FROM lights01 WHERE light_1 = 'green';''' 
      # ),
      # (
      #   'aggregate',
      #   '''SELECT AVG(x.x_min) FROM (SELECT COUNT(frame) as c FROM ligths01 group by light_1)x;''',
      #   '''SELECT AVG(x.x_min) FROM (SELECT COUNT(frame) as c FROM ligths01 group by light_1)x;'''
      # )
    ]

class AggeregateEngineTests(IsolatedAsyncioTestCase):
  def _equality_check(self, aidb_res, gt_res):
    # TO DO
    aidb_res, gt_res = aidb_res[0][0], gt_res[0][0]
    if abs(aidb_res - gt_res) / (gt_res) < 0.05:
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

      print(aidb_res, gt_res, 'aidb_res, gt_res')
      assert self._equality_check(aidb_res, gt_res)

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()