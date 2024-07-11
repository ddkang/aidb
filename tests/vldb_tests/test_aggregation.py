import json
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


SQLITE_URL = 'sqlite+aiosqlite://'

# note: Adjust the AIDB_NUMBER_OF_TEST_RUNS setting for more extensive local testing,
# as it's currently configured for only two runs in GitHub Actions,
# which may not suffice for thorough reliability and functionality checks
_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 10))

DATASET = os.environ.get('DATASET', 'jackson_all')
PORT = int(os.environ.get('PORT', 8000))
TASK = os.environ.get('TASK', 'error')
setup_test_logger(f'aggregation_{DATASET}_{TASK}_{PORT}')


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
    data_dir = os.path.join(dirname, f'data/{DATASET}')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    db_url_list = [SQLITE_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      exact_query_list = []
      approx_query_list = []
      with open(os.path.join(dirname, f'aggregation_queries/{DATASET}/aggregation_{TASK}.sql'), 'r') as f:
        for line in f.readlines():
          exact_query_list.append(line.strip())
      with open(os.path.join(dirname, f'aggregation_queries/{DATASET}/approximate_aggregation_{TASK}.sql'), 'r') as f:
        for line in f.readlines():
          approx_query_list.append(line.strip())

      count_list = [0] * len(approx_query_list)
      k = 0
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir, port=PORT)
      register_inference_services(aidb_engine, data_dir)
      for aidb_query, aggregate_query in zip(approx_query_list, exact_query_list):
        for i in range(_NUMBER_OF_RUNS):
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
          logger.info(f'call: {aidb_engine._config.inference_services["objects00"].infer_one.calls}')
        k+=1
        logger.info(f'Time of runs: {i+1}, Successful count: {count_list}')
      
      # assert sum(count_list) >= len(count_list) * _NUMBER_OF_RUNS - 1
      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
