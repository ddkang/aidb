from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine

DB_URL = 'sqlite+aiosqlite://'


class CachingLogic(IsolatedAsyncioTestCase):

  async def test_num_infer_calls(self):
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
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 300;''',
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 300;'''
      ),
      (
        'full_scan',
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 400;''',
        '''SELECT * FROM objects00 WHERE object_name='car' AND frame < 400;'''
      ),
    ]

    # no service calls before executing query
    assert aidb_engine._config.inference_services["objects00"].infer_one.calls == 0

    calls = [20, 27]
    for index, (query_type, aidb_query, exact_query) in enumerate(queries):
      print(f'Running query {exact_query} in ground truth database')
      # Run the query on the ground truth database
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(exact_query))
        gt_res = gt_res.fetchall()
      # Run the query on the aidb database
      print(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)
      assert len(gt_res) == len(aidb_res)
      # running the same query, so number of inference calls should remain same

      assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index]
      aidb_engine
    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
