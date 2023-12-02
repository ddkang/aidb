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

    queries = [
      (
        'full_scan',
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 100;''',
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 100;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM counts03 WHERE frame >= 10000;''',
        '''SELECT * FROM counts03 WHERE frame >= 10000;''',
      ),
      (
        'full_scan',
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2='green';''',
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2='green';'''
      ),
      (
        'full_scan',
        '''SELECT * FROM objects00 WHERE object_name='car' OR frame < 100;''',
        '''SELECT * FROM objects00 WHERE object_name='car' OR frame < 100;'''
      ),
      (
        'full_scan',
        '''SELECT Avg(x_min) FROM objects00 GROUP BY objects00.object_id;''',
        '''SELECT Avg(x_min) FROM objects00 GROUP BY objects00.object_id;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM objects00 join colors02 on objects00.frame = colors02.frame 
           and objects00.object_id = colors02.object_id;''',

        '''SELECT * FROM objects00 join colors02 on objects00.frame = colors02.frame 
           and objects00.object_id = colors02.object_id;'''
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

  async def test_multi_table_input(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/multi_table_input')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()

    queries = [
      (
        'full_scan',
        '''SELECT tweet_id, sentiment FROM blobs_00;''',
        '''SELECT tweet_id, sentiment FROM blobs_00;''',
      )
    ]

    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

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
