import os
import time
import unittest

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

from multiprocessing import Process

setup_test_logger('full_scan_engine')

DB_URL = "sqlite+aiosqlite://"


# DB_URL = "mysql+aiomysql://aidb:aidb@localhost"
class FullScanEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    def sum_function(a, b, c, d):
      return a + b + c + d

    def multiply(a, b):
      return a * b


    aidb_engine._config.add_user_defined_function('sum_function', sum_function)
    aidb_engine._config.add_user_defined_function('multiply', multiply)

    queries = [
      (
        'full_scan',
        '''
        SELECT objects00.y_max, sum_function(x_min, y_min, x_max, y_max)
        FROM objects00 join colors02 ON objects00.frame = colors02.frame
        WHERE 4000 < multiply(x_min, y_min) AND x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',

        '''SELECT x_min, x_max, y_min, y_max FROM objects00;'''
      ),

    ]

    for query_type, aidb_query, exact_query in queries:
      logger.info(f'Running query {exact_query} in ground truth database')
      # Run the query on the ground truth database
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(exact_query))
        gt_res = gt_res.fetchall()
      # Run the query on the aidb database
      logger.info(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      # TODO: equality check should be implemented
      assert len(gt_res) == len(aidb_res)
    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()