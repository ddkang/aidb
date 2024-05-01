from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger
from aidb.utils.asyncio import asyncio_run
from aidb_utilities.db_setup.clear_cache import clear_ML_cache

setup_test_logger('caching_logic')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

class CachingLogic(IsolatedAsyncioTestCase):

  async def test_num_infer_calls(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    db_url_list = [POSTGRESQL_URL]
    for db_url in db_url_list:
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir)
  
      register_inference_services(aidb_engine, data_dir, batch_supported=False)
  
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
        logger.info(f'Running query {exact_query} in ground truth database')
        # Run the query on the ground truth database
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
        # Run the query on the aidb database
        logger.info(f'Running query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        # running the same query, so number of inference calls should remain same
        # temporarily commenting this out because we no longer call infer_one
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index]
      del gt_engine
      del aidb_engine
    p.terminate()
  
  async def test_cache_clear(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    db_url_list = [POSTGRESQL_URL]
    for db_url in db_url_list:
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir)
  
      register_inference_services(aidb_engine, data_dir, batch_supported=False)
  
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
  
      calls = [[20, 40], [47, 74]]
      # First 300 need 20 calls, 300 - 400 need 7 calls
      for index, (query_type, aidb_query, exact_query) in enumerate(queries):
        logger.info(f'Running query {exact_query} in ground truth database')
        # Run the query on the ground truth database
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
        # Run the query on the aidb database
        logger.info(f'Running initial query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        # running the same query, so number of inference calls should remain same
        # temporarily commenting this out because we no longer call infer_one
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][0], f"Wrong query count: Expected {calls[index][0]}, Actual {aidb_engine._config.inference_services['objects00'].infer_one.calls}"
        logger.info(f'Running cached query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        # cleared cache, so the call count should accumulate as the first run
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][0], f"Wrong query count: Expected {calls[index][0]}, Actual {aidb_engine._config.inference_services['objects00'].infer_one.calls}"
        asyncio_run(clear_ML_cache(aidb_engine))
        logger.info(f'Running uncached query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][1], f"Wrong query count: Expected {calls[index][1]}, Actual {aidb_engine._config.inference_services['objects00'].infer_one.calls}"
      del gt_engine
      del aidb_engine
    p.terminate()

if __name__ == '__main__':
  unittest.main()
