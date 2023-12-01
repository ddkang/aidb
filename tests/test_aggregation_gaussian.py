import logging
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
from tests.utils import setup_gt_and_aidb_engine

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('aggregation_gaussian.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

ground_truth_dir = './tests/data/gaussian/ground_truth'
inference_dir = './tests/data/gaussian/inference'

if not os.path.exists(ground_truth_dir):
    os.makedirs(ground_truth_dir)
if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)

def generate_gaussian_samples(mean, std_dev, num_samples, seed=1234):
  np.random.seed(seed)
  samples = np.random.normal(mean, std_dev, num_samples)
  return samples


def generate_csv_file(mean, std_dev, num_samples):
  samples = generate_gaussian_samples(mean, std_dev, num_samples)
  blobs_00 = pd.DataFrame({'pk_blobs_00.id': list(range(num_samples))})
  blobs_00_csv = os.path.join(ground_truth_dir, 'blobs_00.csv')
  blobs_00.to_csv(blobs_00_csv, index=False)

  gaussian00 = pd.DataFrame({'pk_blobs_00.id': list(range(num_samples)), 'gaussian00.gaussian': samples})
  gaussian00_csv = os.path.join(ground_truth_dir, 'gaussian00.csv')
  gaussian00.to_csv(gaussian00_csv, index=False)

  infer_gaussian00 = pd.DataFrame({'in__blobs_00.id': list(range(num_samples)),
                                   'out__gaussian00.gaussian': samples,
                                   'out__gaussian00.id': list(range(num_samples))})
  infer_gaussian00_csv = os.path.join(inference_dir, 'gaussian00.csv')
  infer_gaussian00.to_csv(infer_gaussian00_csv, index=False)

DB_URL = "sqlite+aiosqlite://"

queries = [
  (
    'approx_aggregate',
    '''SELECT SUM(gaussian) FROM gaussian00 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT SUM(gaussian) FROM gaussian00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT AVG(gaussian) FROM gaussian00 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT AVG(gaussian) FROM gaussian00;'''
  ),
  (
    'approx_aggregate',
    '''SELECT COUNT(gaussian) FROM gaussian00 ERROR_TARGET 5% CONFIDENCE 95%;''',
    '''SELECT COUNT(gaussian) FROM gaussian00;'''
  )
]


class AggeregateEngineTests(IsolatedAsyncioTestCase):
  def _equality_check(self, aidb_res, gt_res, error_target):
    assert len(aidb_res) == len(gt_res)
    for aidb_item, gt_item in zip(aidb_res, gt_res):
      if abs(aidb_item - gt_item) / (gt_item) <= error_target:
        return True
    return False


  async def test_agg_query(self):
    generate_csv_file(100, 100, 1000000)
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/gaussian')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
    register_inference_services(aidb_engine, data_dir)
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
      assert self._equality_check(aidb_res, gt_res, error_target)

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()