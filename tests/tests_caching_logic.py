import os
import time
import unittest
from multiprocessing import Process
from unittest import IsolatedAsyncioTestCase

from sqlalchemy.sql import text

from aidb.utils.asyncio import asyncio_run
from aidb.utils.logger import logger
from tests.inference_service_utils.http_inference_service_setup import \
    run_server
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

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
  
      calls = [[20, 40], [47, 74]]
      # First 300 need 20 calls, 300 to 400 need 7 calls
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
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][0]
        logger.info(f'Running cached query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        # run again, because cache exists, there should be no new calls
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][0]
        asyncio_run(aidb_engine.clear_ml_cache())
        logger.info(f'Running uncached query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        # cleared cache, so should accumulate new calls same as the first call
        assert aidb_engine._config.inference_services["objects00"].infer_one.calls == calls[index][1]
      del gt_engine
      del aidb_engine
    p.terminate()
    p.join()

  async def test_only_one_service_deleted(self):
    '''
    Testing whether cache for other service remains when only one service is deleted.
    Do query on two different services first. Then delete cache for one service.
    Finally do query on these services again and check whether the call count is correct.
    '''
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
          '''SELECT * FROM lights01 WHERE light_1='red' AND frame < 300;''',
          '''SELECT * FROM lights01 WHERE light_1='red' AND frame < 300;'''
        ),
        (
          'full_scan',
          '''SELECT * FROM counts03 WHERE count = 1 AND frame < 300;''',
          '''SELECT * FROM counts03 WHERE count = 1 AND frame < 300;'''
        ),
      ]
  
      # no service calls before executing query
      # all the infer_one call use the same counter, so checking only one of them should be enough
      assert aidb_engine._config.inference_services["counts03"].infer_one.calls == 0
      
      calls = [[20, 40], [60, 60]]
      # each query calls 20 inference.
      # For the first two queries, all the inference is needed.
      # The third query need to infer again but the fourth don't.
      for index, (query_type, aidb_query, exact_query) in enumerate(queries):
        # Run the query on the aidb database
        logger.info(f'Running query {exact_query} in ground truth database')
        # Run the query on the ground truth database
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
        logger.info(f'Running initial query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        assert aidb_engine._config.inference_services["counts03"].infer_one.calls == calls[0][index]

      asyncio_run(aidb_engine.clear_ml_cache(["lights01"]))
      
      for index, (query_type, aidb_query, exact_query) in enumerate(queries):
        # Run the query on the aidb database
        logger.info(f'Running query {exact_query} in ground truth database')
        # Run the query on the ground truth database
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
        logger.info(f'Running query {aidb_query} in aidb database after cache deleted')
        aidb_res = aidb_engine.execute(aidb_query)
        assert len(gt_res) == len(aidb_res)
        assert aidb_engine._config.inference_services["counts03"].infer_one.calls == calls[1][index]
      
      del gt_engine
      del aidb_engine
    p.terminate()
    p.join()

if __name__ == '__main__':
  unittest.main()
