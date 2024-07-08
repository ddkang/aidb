from multiprocessing import Process
import numpy as np
import os
import pandas as pd
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

setup_test_logger('aggregation_gaussian')

ground_truth_dir = './tests/vldb_tests/data/gaussian/ground_truth'
inference_dir = './tests/vldb_tests/data/gaussian/inference'

if not os.path.exists(ground_truth_dir):
    os.makedirs(ground_truth_dir)
if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)

_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 1))

def generate_gaussian_samples(mean, std_dev, num_samples, seed=1234):
  # np.random.seed(seed)
  samples = np.random.normal(mean, std_dev, num_samples)
  return samples


def generate_csv_file(mean, std_dev, num_samples):
  random_values_np = np.random.randint(0, 10, size=num_samples)
  repeated_ids = [i for i, j in zip(list(range(num_samples)), random_values_np) for _ in range(j)]
  output_ids = [j for i in random_values_np for j in range(i)]

  samples = generate_gaussian_samples(mean, std_dev, len(repeated_ids))

  blobs_00 = pd.DataFrame({'pk_blobs_00.id': list(range(num_samples))})
  blobs_00_csv = os.path.join(ground_truth_dir, 'blobs_00.csv')
  blobs_00.to_csv(blobs_00_csv, index=False)

  gaussian00 = pd.DataFrame({'pk_blobs_00.id': repeated_ids, 'pk_gaussian00.output_id':output_ids, 'gaussian00.gaussian_value': samples})
  gaussian00_csv = os.path.join(ground_truth_dir, 'gaussian00.csv')
  gaussian00.to_csv(gaussian00_csv, index=False)

  infer_gaussian00 = pd.DataFrame({'in__blobs_00.id': repeated_ids,
                                   'out__gaussian00.output_id': output_ids,
                                   'out__gaussian00.gaussian_value': samples,
                                   'out__gaussian00.id': repeated_ids})
  infer_gaussian00_csv = os.path.join(inference_dir, 'gaussian00.csv')
  infer_gaussian00.to_csv(infer_gaussian00_csv, index=False)

DB_URL = "sqlite+aiosqlite://"

queries = [
  (
    'approx_aggregate',
    '''SELECT SUM(gaussian_value) FROM gaussian00 WHERE gaussian_value > 1000 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT SUM(gaussian_value) FROM gaussian00 WHERE gaussian_value > 1000;'''
  )
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
    generate_csv_file(0, 1000, 1000000)
    count_list = [0] * len(queries)
    for i in range(_NUMBER_OF_RUNS):
      dirname = os.path.dirname(__file__)
      data_dir = os.path.join(dirname, 'data/gaussian')

      p = Process(target=run_server, args=[str(data_dir)])
      p.start()
      time.sleep(1)
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
      register_inference_services(aidb_engine, data_dir)
      k = 0
      for query_type, aidb_query, aggregate_query in queries:
        logger.info(f'Running query {aggregate_query} in ground truth database')
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(aggregate_query))
          gt_res = gt_res.fetchall()[0]
        logger.info(f'Running query {aidb_query} in aidb database')
        aidb_res = aidb_engine.execute(aidb_query)[0]
        logger.info(f'aidb_res: {aidb_res}, gt_res: {gt_res}')
        error_target = Query(aidb_query, aidb_engine._config).error_target
        if error_target is None: error_target = 0
        if self._equality_check(aidb_res, gt_res, error_target):
          count_list[k] += 1
        k += 1

      logger.info(f'Time of runs: {i + 1}, Successful count: {count_list}')
      del gt_engine
      del aidb_engine
      p.terminate()


if __name__ == '__main__':
  unittest.main()
