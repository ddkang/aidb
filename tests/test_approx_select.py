import logging
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
from tests.utils import setup_gt_and_aidb_engine

logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('approx_select.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

DB_URL = 'sqlite+aiosqlite://'
DATA_SET = 'law'
BUDGET = 5000
RECALL_TARGET = 90

# DB_URL = 'mysql+aiomysql://aidb:aidb@localhost'
class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, f'data/{DATA_SET}')
    p = mp.Process(target=run_server, args=[str(data_dir)])
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

    count = 0
    for i in range(100):
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir, tasti)

      register_inference_services(aidb_engine, data_dir)
      queries = [
        (
          f'''SELECT entity_id FROM entities00 where type LIKE 'PERSON'
              RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;''',
          '''SELECT entity_id FROM entities00 where type LIKE 'PERSON';'''
        ),
      ]

      for aidb_query, exact_query in queries:
        logger.info(f'Running query {aidb_query} in approx select engine')
        seed = (mp.current_process().pid * np.random.randint(100000, size=1)[0]) % (2**32 - 1)
        aidb_res = aidb_engine.execute(aidb_query, seed=seed)

        logger.info(f'Running query {exact_query} in ground truth database')
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()

        if len(aidb_res) / len(gt_res) > RECALL_TARGET / 100:
          count += 1

      logger.info(f'AIDB_res: {len(aidb_res)}, gt_res:{len(gt_res)}, Recall: {len(aidb_res) / len(gt_res)},'
                   f' Times of trial:{i + 1}, Count: {count}')

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()