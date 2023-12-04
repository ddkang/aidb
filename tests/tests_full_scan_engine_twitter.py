import os
import time
import unittest

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine

from multiprocessing import Process

DB_URL = "sqlite+aiosqlite://"


class FullScanEngineTwitterTests(IsolatedAsyncioTestCase):

  async def test_twitter(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/twitter_all')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    queries = [
      (
        'full_scan',
        '''SELECT * FROM hate01 WHERE ishate=1 and tweet_id < 500;''',
        '''SELECT * FROM hate01 WHERE ishate=1 and tweet_id < 500;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM sentiment02 WHERE sentiment=-1 and tweet_id < 1000;''',
        '''SELECT * FROM sentiment02 WHERE sentiment=-1 and tweet_id < 1000;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM topic03 WHERE topic='sports_&_gaming' and tweet_id < 1000;''',
        '''SELECT * FROM topic03 WHERE topic='sports_&_gaming' and tweet_id < 1000;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM entities00 WHERE type='person' and tweet_id < 2000;''',
        '''SELECT * FROM entities00 WHERE type='person' and tweet_id < 2000;'''
      ),
    ]

    for query_type, aidb_query, exact_query in queries:
      print(f'Running query {exact_query} in ground truth database')
      # Run the query on the ground truth database
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(exact_query))
        gt_res = gt_res.fetchall()
      # Run the query on the aidb database
      print(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      print("Length ground truth - ", len(gt_res))
      assert len(gt_res) == len(aidb_res)
    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
