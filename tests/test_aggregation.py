from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine


DB_URL = "sqlite+aiosqlite://"

queries = [
  (
    'approx_aggregate',
    '''SELECT SUM(x_min) FROM objects00 ERROR_TARGET 20% CONFIDENCE 95;''',
    '''SELECT SUM(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(y_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT SUM(y_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(x_min) FROM objects00 ERROR_TARGET 5% CONFIDENCE 95;''',
    '''SELECT COUNT(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_max) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(x_max) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(y_max) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(y_max) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(y_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(y_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE y_max < 900 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE y_max < 900;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min < 700 ERROR_TARGET 10% CONFIDENCE 95;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min < 700;'''
  ),
]


class AggeregateEngineTests(IsolatedAsyncioTestCase):
  def _equality_check(self, aidb_res, gt_res, error_target):
    assert len(aidb_res) == len(gt_res)
    for aidb_item, gt_item in zip(aidb_res, gt_res):
      if abs(aidb_item - gt_item) / (gt_item) <= error_target:
        return True
    return False


  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson_all')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
    register_inference_services(aidb_engine, data_dir)
    for query_type, aidb_query, aggregate_query in queries:
      print(f'Running query {aggregate_query} in ground truth database')
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(aggregate_query))
        gt_res = gt_res.fetchall()[0]
      print(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)[0]
      print(f'aidb_res: {aidb_res}, gt_res: {gt_res}')
      error_target = Query(aidb_query, aidb_engine._config).error_target
      if error_target is None: error_target = 0
      assert self._equality_check(aidb_res, gt_res, error_target)

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()