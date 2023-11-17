from multiprocessing import Process
import numpy as np
import os

import pandas as pd
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase

from aidb.vector_database.tasti import Tasti
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.tasti_test.tasti_test import TastiTests, VectorDatabaseType
from tests.utils import setup_gt_and_aidb_engine

DB_URL = 'sqlite+aiosqlite://'
DATA_SET = 'law'
# DB_URL = 'mysql+aiomysql://aidb:aidb@localhost'
class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, f'data/{DATA_SET}')
    p = Process(target=run_server, args=[str(data_dir)])
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
    tasti = Tasti(index_name, user_database, 5000)

    count = 0
    for i in range(10):
      gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir, tasti)


      register_inference_services(aidb_engine, data_dir)
      queries = [
        # (
        #   '''SELECT * FROM objects00 WHERE x_min > 1000 RECALL_TARGET 40 CONFIDENCE 95% BUDGET 2000;''',
        #   '''SELECT * FROM objects00 WHERE x_min > 1000;'''
        # ),
        (
          '''SELECT blob_id, entity_id FROM entities00 where type LIKE 'PERSON' RECALL_TARGET 40 CONFIDENCE 95% BUDGET 2000;''',
          '''SELECT blob_id, entity_id FROM entities00 where type LIKE 'PERSON';'''
        ),
      ]

      for aidb_query, exact_query in queries:
        print(f'Running query {aidb_query} in limit engine')
        aidb_res = aidb_engine.execute(aidb_query)

        print(f'Running query {exact_query} in ground truth database')
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
        print(len(aidb_res), len(gt_res), len(aidb_res) / len(gt_res))
        if len(aidb_res) / len(gt_res) > 0.4:
          count += 1
        print(count, i+1)

      del gt_engine
      del aidb_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()