from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

setup_test_logger('exact_aggregation')

DB_URL = "sqlite+aiosqlite://"

queries = [
  (
    'exact_aggregate',
    '''SELECT COUNT(*) FROM objects00;''',
    '''SELECT COUNT(*) FROM objects00;'''
  ),
  (
    'exact_aggregate',
    '''SELECT SUM(x_max) FROM objects00;''',
    '''SELECT SUM(x_max) FROM objects00;'''
  ),
  (
    'exact_aggregate',
    '''SELECT COUNT(*) FROM colors02 WHERE color='black';''',
    '''SELECT COUNT(*) FROM colors02 WHERE color='black';'''
  ),
]


class ExactAggeregateTests(IsolatedAsyncioTestCase):
  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
    register_inference_services(aidb_engine, data_dir)
    for query_type, aidb_query, aggregate_query in queries:
      logger.info(f'Running query {aggregate_query} in ground truth database')
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(aggregate_query))
        gt_res = gt_res.fetchall()[0]
      logger.info(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)[0]
      logger.info(f'aidb_res: {aidb_res}, gt_res: {gt_res}')
      assert aidb_res[0] == gt_res[0]

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()