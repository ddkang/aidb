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


# DB_URL = "mysql+aiomysql://aidb:aidb@localhost"
class NestedQueryTests(IsolatedAsyncioTestCase):

  async def test_nested_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    queries = [
      # subquery is on meta table. always satisfied
      (
        'full_scan',
        '''SELECT * FROM objects00 WHERE object_name='car' AND objects00.frame < 
              (SELECT AVG(blobs_00.frame) FROM blobs_00) OR objects00.frame NOT IN (1, 2, 3);''',
        '''SELECT * FROM objects00 WHERE object_name='car' AND objects00.frame < 
              (SELECT AVG(blobs_00.frame) FROM blobs_00) OR objects00.frame NOT IN (1, 2, 3);'''
      ),

      # 2-place predicates. subquery is not satisfied until objects00 is filled
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.frame >= 10000 AND colors02.object_id > 
              (SELECT AVG(objects00.object_id) FROM objects00);''',
        '''SELECT * FROM colors02 WHERE colors02.frame >= 10000 AND colors02.object_id > 
              (SELECT AVG(objects00.object_id) FROM objects00);'''
      ),
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.frame >= 10000;''',
        '''SELECT * FROM colors02 WHERE colors02.frame >= 10000;'''
      ),
      # multiple sub-queries
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.object_id > (SELECT AVG(objects00.object_id) FROM objects00) 
              OR colors02.object_id > (SELECT AVG(colors02.object_id) FROM colors02);''',
        '''SELECT * FROM colors02 WHERE colors02.object_id > (SELECT AVG(objects00.object_id) FROM objects00) 
              OR colors02.object_id > (SELECT AVG(colors02.object_id) FROM colors02);''',
      ),

      # nested sub-queries. predicate clause is not satisfied until all inference is complete
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.object_id > (SELECT AVG(objects00.object_id) FROM objects00 
              WHERE objects00.y_max < (SELECT AVG(colors02.object_id) FROM colors02));''',
        '''SELECT * FROM colors02 WHERE colors02.object_id > (SELECT AVG(objects00.object_id) FROM objects00 
              WHERE objects00.y_max < (SELECT AVG(colors02.object_id) FROM colors02));''',
      ),

      # where-in predicate. frame is not satisfied until objects00 is filled
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT objects00.object_id FROM objects00 
              WHERE objects00.frame < 1000);''',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT objects00.object_id FROM objects00 
              WHERE objects00.frame < 1000);'''
      ),

      # where-in predicate. x_min is not satisfied until objects00 is filled
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT objects00.object_id FROM objects00 
              WHERE objects00.x_min < 364);''',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT objects00.object_id FROM objects00 
              WHERE objects00.x_min < 364);'''
      ),

      # where-in predicate. color is not satisfied until colors02 is filled
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT colors02.object_id FROM colors02 
              WHERE colors02.color = 'grayish_blue');''',
        '''SELECT * FROM colors02 WHERE colors02.object_id IN (SELECT colors02.object_id FROM colors02 
              WHERE colors02.color = 'grayish_blue');'''
      ),

      # where-in predicate with literals
      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE object_id IN (500,501,502);''',
        '''SELECT * FROM colors02 WHERE object_id IN (500,501,502);'''
      ),

      (
        'full_scan',
        '''SELECT * FROM colors02 WHERE 500 < object_id < 1000;''',
        '''SELECT * FROM colors02 WHERE 500 < object_id < 1000;'''
      )
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

      assert len(gt_res) == len(aidb_res)

      # Sort results
      gt_res = sorted(gt_res)
      aidb_res = sorted(aidb_res)

      for i in range(len(gt_res)):
        assert gt_res[i] == aidb_res[i]
    del gt_engine
    p.terminate()

if __name__ == '__main__':
  unittest.main()