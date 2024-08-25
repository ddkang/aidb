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

setup_test_logger('limit_engine')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

DATA_SET = 'twitter'
BUDGET = 5000

class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, f'data/{DATA_SET}')
    p = Process(target=run_server, args=[str(data_dir), 8050])
    p.start()
    time.sleep(1)

    # vector database configuration
    index_path = './'
    index_name = DATA_SET
    embedding = np.load(f'./tests/data/embedding/{DATA_SET}_embeddings.npy')
    embedding_df = pd.DataFrame({'id': range(embedding.shape[0]), 'values': embedding.tolist()})

    embedding_dim = embedding.shape[1]
    user_database = FaissVectorDatabase(index_path)
    user_database.create_index(index_name, embedding_dim, recreate_index=True)
    user_database.insert_data(index_name, embedding_df)
    seed = mp.current_process().pid
    tasti = Tasti(index_name, user_database, BUDGET, seed=seed)

    queries = [
      (
        '''SELECT * FROM entity00 join sentiment01 on entity00.tweet_id = sentiment01.tweet_id WHERE type LIKE 'ORG' AND label LIKE 'positive';''',
        '''SELECT * FROM entity00 join sentiment01 on entity00.tweet_id = sentiment01.tweet_id WHERE type LIKE 'ORG' AND label LIKE 'positive';'''
      )
    ]
    db_url_list = [ SQLITE_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      for aidb_query, exact_query in queries:
        logger.info(f'Running query {aidb_query} in limit engine')

        gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir, tasti, port=8050)
        cost_dict = {'entity00': 15, 'sentiment01': 2.25}
        register_inference_services(aidb_engine, data_dir, port=8050, cost_dict=cost_dict)
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
        assert len(set(aidb_res) - set(gt_res)) == 0

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()
