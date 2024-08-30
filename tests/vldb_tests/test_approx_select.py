import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import time
import unittest

from sqlalchemy.sql import text
from unittest import IsolatedAsyncioTestCase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.tasti import Tasti
from aidb.utils.logger import logger
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger


SQLITE_URL = 'sqlite+aiosqlite://'

_NUMBER_OF_RUNS = int(os.environ.get('AIDB_NUMBER_OF_TEST_RUNS', 10))

DATASET = os.environ.get('DATASET', 'law')
RECALL_TARGET = int(os.environ.get('RECALL_TARGET', 90))
PORT = int(os.environ.get('PORT', 8000))
TASK = os.environ.get('TASK', 'error')
BUDGET = int(os.environ.get('BUDGET', 5000))
setup_test_logger(f'approx_select_{DATASET}_{RECALL_TARGET}_{PORT}')


class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, f'data/{DATASET}')
    p = mp.Process(target=run_server, args=[str(data_dir), PORT])
    p.start()
    time.sleep(1)

    # vector database configuration
    index_path = './'
    index_name = f'{DATASET}_{RECALL_TARGET}_{PORT}'
    embedding = np.load(f'./tests/vldb_tests/data/embedding/{DATASET}_embeddings.npy')
    embedding_df = pd.DataFrame({'id': range(embedding.shape[0]), 'values': embedding.tolist()})

    embedding_dim = embedding.shape[1]
    user_database = FaissVectorDatabase(index_path)
    user_database.create_index(index_name, embedding_dim, recreate_index=True)
    user_database.insert_data(index_name, embedding_df)
    seed = mp.current_process().pid
    tasti = Tasti(index_name, user_database, BUDGET, seed=seed)

    queries = [
      # (
      #   f'''SELECT entity_id FROM entity00 where type LIKE 'PERSON'
      #               RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;''',
      #   '''SELECT entity_id FROM entity00 where type LIKE 'PERSON';'''
      # )
      (
        f'''SELECT frame FROM objects00 where y_min > 0
                    RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;''',
        '''SELECT frame FROM objects00 where y_min > 0;'''
      ),
    ]
    queries = []
    with open(os.path.join(dirname, f'aggregation_queries/{DATASET}/approx_select.sql'), 'r') as f:
      for line in f.readlines():
        queries.append((f'{line} RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%', line))
        
    db_url_list = [SQLITE_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      count_list = [0] * len(queries)
      
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir, tasti, port=PORT)
      register_inference_services(aidb_engine, data_dir, port=PORT)
      for i in range(_NUMBER_OF_RUNS):
        k = 0
        for aidb_query, exact_query in queries:
          logger.info(f'Running query {aidb_query} in approx select engine')
          seed = (mp.current_process().pid * np.random.randint(100000, size=1)[0]) % (2**32 - 1)
          aidb_res = aidb_engine.execute(aidb_query, __seed=seed)

          logger.info(f'Running query {exact_query} in ground truth database')
          try:
            async with gt_engine.begin() as conn:
              gt_res = await conn.execute(text(exact_query))
              gt_res = gt_res.fetchall()
          finally:
            await gt_engine.dispose()

          if len(aidb_res) / len(gt_res) > RECALL_TARGET / 100:
            count_list[k] += 1
          k += 1
          logger.info(f'AIDB_res: {len(aidb_res)}, gt_res:{len(gt_res)}, Recall: {len(aidb_res) / len(gt_res)},'
                       f' Times of trial:{i + 1}, Count: {count_list}')

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()