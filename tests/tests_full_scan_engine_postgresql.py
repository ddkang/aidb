import os
import time
import unittest

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

from aidb.utils.logger import logger
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

from multiprocessing import Process

setup_test_logger('full_scan_engine')

DB_URL = "postgresql+asyncpg://user:testaidb@localhost:5432"

# DB_URL = "mysql+aiomysql://root:testaidb@localhost:3306"
class FullScanEngineTests(IsolatedAsyncioTestCase):

  async def test_jackson_number_objects(self):

    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(3)
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    queries = [
      (
        'full_scan',
        '''SELECT * FROM counts03;''',
        '''SELECT * FROM counts03;''',
      )

    ]

    for query_type, aidb_query, exact_query in queries:
      logger.info(f'Running query {exact_query} in ground truth database')
      # Run the query on the ground truth database
      try:
        async with gt_engine.begin() as conn:
          gt_res = await conn.execute(text(exact_query))
          gt_res = gt_res.fetchall()
      finally:
        await gt_engine.dispose()

      # Run the query on the aidb database
      logger.info(f'Running query {aidb_query} in aidb database')
      aidb_res = aidb_engine.execute(aidb_query)

      # TODO: equality check should be implemented
      assert len(gt_res) == len(aidb_res)
    p.terminate()


if __name__ == '__main__':
  unittest.main()