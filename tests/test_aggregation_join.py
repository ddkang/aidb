import os
import time
import unittest
from decimal import Decimal
from multiprocessing import Process
from unittest import IsolatedAsyncioTestCase

import pandas as pd
from sqlalchemy.sql import text

from aidb.inference.bound_inference_service import CachedBoundInferenceService
from aidb.query.query import Query
from aidb.utils.logger import logger
from tests.inference_service_utils.http_inference_service_setup import \
    run_server
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

setup_test_logger('aggregation_join')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 1))

def inference(inference_service: CachedBoundInferenceService, input_df: pd.DataFrame):
  input_df.columns = inference_service.binding.input_columns
  outputs = inference_service.service.infer_batch(input_df)
  return outputs


queries = [
  (
    'approx_aggregate',
    '''SELECT COUNT(*) FROM blobs_00 CROSS JOIN blobs_01 WHERE match_inference(blobs_00.img_id, blobs_01.text_id) = TRUE
           AND blobs_00.img_id > 10000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(match00.img_id) FROM match00 WHERE match00.img_id > 10000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(*), AVG(blobs_00.img_id) FROM blobs_00 CROSS JOIN blobs_01 WHERE match_inference(blobs_00.img_id, blobs_01.text_id) = TRUE 
           AND blobs_00.img_id > 10000 ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(match00.img_id), AVG(match00.img_id) FROM match00 WHERE match00.img_id > 10000;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(*) FROM blobs_00 CROSS JOIN blobs_01 WHERE match_inference(blobs_00.img_id, blobs_01.text_id) = TRUE  
           ERROR_TARGET 10% CONFIDENCE 95%;''',
    '''SELECT COUNT(match00.img_id) FROM match00;'''
  ),
]


class AggregateJoinEngineTests(IsolatedAsyncioTestCase):
  def add_user_defined_function(self, aidb_engine):
    async def async_duplicate_inference(input_df):
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'duplicate01':
          return inference(service, input_df)


    async def async_match_inference(input_df):
      for service in aidb_engine._config.inference_bindings:
        if service.service.name == 'match00':
          return inference(service, input_df)

    aidb_engine._config.add_user_defined_function('duplicate_inference', async_duplicate_inference)
    aidb_engine._config.add_user_defined_function('match_inference', async_match_inference)

  def _equality_check(self, aidb_res, gt_res, error_target):
    assert len(aidb_res) == len(gt_res)
    error_rate_list = []
    valid_estimation = True
    for aidb_item, gt_item in zip(aidb_res, gt_res):
      if isinstance(gt_item, Decimal):
        gt_item = float(gt_item)
      relative_diff = abs(aidb_item - gt_item) / (gt_item)
      error_rate_list.append(relative_diff * 100)
      if relative_diff > error_target:
        valid_estimation = False
    logger.info(f'Error rate (%) for approximate aggregation query: {error_rate_list}')
    return valid_estimation


  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/flickr30k_join')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    db_url_list = [POSTGRESQL_URL, SQLITE_URL, MYSQL_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      count_list = [0] * len(queries)
      for i in range(_NUMBER_OF_RUNS):
        gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir)
        self.add_user_defined_function(aidb_engine)

        register_inference_services(aidb_engine, data_dir)
        k = 0
        for query_type, aidb_query, aggregate_query in queries:
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
        del gt_engine
        del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
