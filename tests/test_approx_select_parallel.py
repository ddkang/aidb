import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from sqlalchemy.sql import text
import time

from aidb.utils.asyncio import asyncio_run
from aidb.vector_database.tasti import Tasti
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler = logging.FileHandler('approx_selection.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


DB_URL = 'sqlite+aiosqlite://'
DATA_SET = 'law'
BUDGET = 5000
RECALL_TARGET = 90
# DB_URL = 'mysql+aiomysql://aidb:aidb@localhost'

async def test_jackson_number_objects(i):
  dirname = os.path.dirname(__file__)
  data_dir = os.path.join(dirname, f'data/{DATA_SET}')
  p = mp.Process(target=run_server, args=[str(data_dir), i])
  p.start()
  time.sleep(1)

  # vector database configuration
  index_path = './'
  index_name = f'{DATA_SET}_{i}'
  embedding = np.load(f'./tests/data/embedding/{DATA_SET}_embeddings.npy')
  embedding_df = pd.DataFrame({'id': range(embedding.shape[0]), 'values': embedding.tolist()})

  embedding_dim = embedding.shape[1]
  user_database = FaissVectorDatabase(index_path)
  user_database.create_index(index_name, embedding_dim, recreate_index=True)
  user_database.insert_data(index_name, embedding_df)
  seed = mp.current_process().pid
  tasti = Tasti(index_name, user_database, BUDGET, seed=seed)

  count = 0
  gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir, tasti, i)


  register_inference_services(aidb_engine, data_dir, i)
  queries = [
    (
      f'''SELECT entity_id FROM entities00 where type LIKE 'PERSON'
                RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;''',
      '''SELECT entity_id FROM entities00 where type LIKE 'PERSON';'''
    ),
  ]

  for aidb_query, exact_query in queries:
    print(f'Running query {aidb_query} in limit engine')
    aidb_res = aidb_engine.execute(aidb_query)

    print(f'Running query {exact_query} in ground truth database')
    async with gt_engine.begin() as conn:
      gt_res = await conn.execute(text(exact_query))
      gt_res = gt_res.fetchall()

    if len(aidb_res) / len(gt_res) > RECALL_TARGET / 100:
      count += 1

  logging.info(f'aidb: {len(aidb_res)}, gt_res:{len(gt_res)}, Recall: {len(aidb_res) / len(gt_res)},'
               f' port:{i}, Count: {count}')
  del gt_engine
  del aidb_engine
  p.terminate()


def run_async_outer_process(port):
  try:
    asyncio_run(test_jackson_number_objects(port))
  except Exception as e:
    raise e


if __name__ == '__main__':
  processes = []
  ports = range(8000, 8010)

  for port in ports:
    # Create a non-daemonic process for each port
    process = mp.Process(target=run_async_outer_process, args=(port,))
    processes.append(process)
    process.start()

  # Wait for all processes to complete
  for process in processes:
    process.join()