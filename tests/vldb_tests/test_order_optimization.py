from aidb.utils.order_optimization_utils import get_currently_supported_filtering_predicates_for_ordering
from aidb.query.query import Query

from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase
import numpy as np
import pandas as pd
import multiprocessing as mp

from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.tasti_test.tasti_test import TastiTests, VectorDatabaseType
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.tasti import Tasti


POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 10))

DATASET = os.environ.get('DATASET', 'twitter')
PORT = int(os.environ.get('PORT', 8000))
BUDGET = int(os.environ.get('BUDGET', 5000))
setup_test_logger(f'order_optimization_{DATASET}_{PORT}')


class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, f'data/{DATASET}')
    p = Process(target=run_server, args=[str(data_dir), PORT])
    p.start()
    time.sleep(1)

    # vector database configuration
    index_path = './'
    index_name = DATASET
    embedding = np.load(f'./tests/vldb_tests/data/embedding/{DATASET}_embeddings.npy')
    embedding_df = pd.DataFrame({'id': range(embedding.shape[0]), 'values': embedding.tolist()})

    embedding_dim = embedding.shape[1]
    user_database = FaissVectorDatabase(index_path)
    user_database.create_index(index_name, embedding_dim, recreate_index=True)
    user_database.insert_data(index_name, embedding_df)
    seed = mp.current_process().pid
    tasti = Tasti(index_name, user_database, BUDGET, seed=seed)

    queries = []
    with open(os.path.join(dirname, f'aggregation_queries/{DATASET}/exact.sql'), 'r') as f:
      for line in f.readlines():
        queries.append((line, line))
    db_url_list = [ SQLITE_URL]
    cost_dict_all = {
      'twitter': {'entity00': 0.0010, 'sentiment01': 0.0010}, 
      'arxiv': {'arxiv00': 1.5/1000, 'sentiment01': 0.0010},
      'law': {'entity00': 0.0010, 'sentiment01': 0.0010},
      'jackson_all': {'objects00': 2.25 / 1000, 'color01':1.5/1000, 'lights01': 1.5/1000}
    }
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      for aidb_query, exact_query in queries:
        logger.info(f'Running query {aidb_query} in full scan engine to test the optimization of ML ordering')

        gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir, tasti, port=PORT)
        cost_dict = cost_dict_all[DATASET]
        register_inference_services(aidb_engine, data_dir, port=PORT, cost_dict=cost_dict)
        aidb_res = aidb_engine.execute(aidb_query)

        logger.info(f'Running query {exact_query} in ground truth database')
        try:
          async with gt_engine.begin() as conn:
            gt_res = await conn.execute(text(exact_query))
            gt_res = gt_res.fetchall()
        finally:
          await gt_engine.dispose()

        logger.info(f'There are {len(aidb_res)} elements in limit engine results '
              f'and {len(gt_res)} elements in ground truth results')
        logger.info(f'difference: {set(aidb_res) - set(gt_res)}')
        assert len(set(aidb_res) - set(gt_res)) == 0

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
