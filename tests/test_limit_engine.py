from multiprocessing import Process
import os
from sqlalchemy.sql import text
import time
import unittest
from unittest import IsolatedAsyncioTestCase


from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.tasti_test.tasti_test import TastiTests, VectorDatabaseType
from tests.utils import setup_gt_and_aidb_engine

DB_URL = 'sqlite+aiosqlite://'

# DB_URL = 'mysql+aiomysql://aidb:aidb@localhost'
class LimitEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)

    # vector database configuration
    index_name = 'tasti'
    data_size = 1000
    embedding_dim = 128
    nb_buckets = 100
    vector_database_type = VectorDatabaseType.FAISS.value

    tasti_test = TastiTests(index_name, data_size, embedding_dim, nb_buckets, vector_database_type, index_path='./')
    tasti_index = tasti_test.tasti
    blob_mapping_table = 'blobs_mapping'

    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir, tasti_index, blob_mapping_table)


    register_inference_services(aidb_engine, data_dir)
    queries = [
      (
        '''SELECT * FROM colors02 WHERE frame >= 1000 and colors02.color = 'black' LIMIT 100;''',
        '''SELECT * FROM colors02 WHERE frame >= 1000 and colors02.color = 'black';'''
      ),
      (
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2 = 'green' LIMIT 100;''',
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2 = 'green';'''
      ),
      (
        '''SELECT * FROM objects00 WHERE object_name = 'car' OR frame < 100 LIMIT 100;''',
        '''SELECT * FROM objects00 WHERE object_name = 'car' OR frame < 100;'''
      )
    ]

    for aidb_query, exact_query in queries:
      print(f'Running query {aidb_query} in limit engine')
      aidb_res = aidb_engine.execute(aidb_query)

      print(f'Running query {exact_query} in ground truth database')
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(exact_query))
        gt_res = gt_res.fetchall()

      print(f'There are {len(aidb_res)} elements in limit engine results '
            f'and {len(gt_res)} elements in ground truth results')
      assert len(set(aidb_res) - set(gt_res)) == 0

    del gt_engine
    p.terminate()


if __name__ == '__main__':
  unittest.main()