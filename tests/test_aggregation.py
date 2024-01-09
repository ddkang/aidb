import random
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

setup_test_logger('aggregation')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 100))

queries = [
  (
    'approx_aggregate',
    '''SELECT SUM(x_min), COUNT(frame) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min), COUNT(frame) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min), COUNT(*) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min), COUNT(*) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(x_min), SUM(y_max), AVG(x_max), COUNT(*) FROM objects00
           WHERE y_min > 500 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min), SUM(y_max), AVG(x_max), COUNT(*) FROM objects00 WHERE y_min > 500;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(x_min), SUM(y_max), SUM(x_max), SUM(y_min) FROM objects00
           WHERE x_min < 1000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min), SUM(y_max), SUM(x_max), SUM(y_min) FROM objects00 WHERE x_min < 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min), SUM(y_max), AVG(x_max), SUM(y_min) FROM objects00
           WHERE frame > 100000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min), SUM(y_max), AVG(x_max), SUM(y_min) FROM objects00
           WHERE frame > 100000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(x_min), SUM(y_max), COUNT(x_max), AVG(y_min) FROM objects00
           WHERE x_min > 700 AND y_min > 700 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(x_min), SUM(y_max), COUNT(x_max), AVG(y_min) FROM objects00 WHERE x_min > 700 AND y_min > 700;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(x_min) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(x_min) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(x_min) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(y_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(y_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_max) FROM objects00 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(x_max) FROM objects00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min > 1000 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min > 1000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE y_max < 900 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE y_max < 900;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min < 700 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE x_min < 700;'''
  ),
  (
    'approx_aggregate',
    '''SELECT SUM(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT SUM(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00);'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT AVG(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00);'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(x_min) FROM objects00 WHERE frame > (SELECT AVG(frame) FROM blobs_00);'''
  ),
]


class AggeregateEngineTests(IsolatedAsyncioTestCase):
  def _equality_check(self, aidb_res, gt_res, error_target):
    assert len(aidb_res) == len(gt_res)
    error_rate_list = []
    valid_estimation = True
    for aidb_item, gt_item in zip(aidb_res, gt_res):
      relative_diff = abs(aidb_item - gt_item) / (gt_item)
      error_rate_list.append(relative_diff * 100)
      if relative_diff > error_target:
        valid_estimation = False
    logger.info(f'Error rate (%) for approximate aggregation query: {error_rate_list}')
    return valid_estimation


  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson_all')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    db_url_list = [MYSQL_URL, POSTGRESQL_URL, SQLITE_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      # randomly choose 3 queries to test PostgreSQL and MySQL
      if dialect != 'sqlite':
        selected_queries = random.sample(queries, 3)
      else:
        selected_queries = queries
      count_list = [0] * len(selected_queries)
      for i in range(_NUMBER_OF_RUNS):
        gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir)
        register_inference_services(aidb_engine, data_dir)
        k = 0
        for query_type, aidb_query, aggregate_query in selected_queries:
          logger.info(f'Running query {aggregate_query} in ground truth database')
          try:
            async with gt_engine.begin() as conn:
              gt_res = await conn.execute(text(aggregate_query))
              gt_res = gt_res.fetchall()[0]
          finally:
            await gt_engine.dispose()
          logger.info(f'Running query {aidb_query} in aidb database')
          aidb_res = aidb_engine.execute(aidb_query)[0]
          logger.info(f'aidb_res: {aidb_res}, gt_res: {gt_res}')
          error_target = Query(aidb_query, aidb_engine._config).error_target
          if error_target is None: error_target = 0
          if self._equality_check(aidb_res, gt_res, error_target):
            count_list[k] += 1
          k+=1
        logger.info(f'Time of runs: {i+1}, Successful count: {count_list}')
        assert sum(count_list) >= len(count_list) * num_runs - 1
        del gt_engine
        del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
