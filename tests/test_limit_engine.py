import os
import time
import unittest
from multiprocessing import Process
from unittest import IsolatedAsyncioTestCase

import pandas as pd
from sqlalchemy.sql import text

from aidb.utils.constants import table_name_for_rep_and_topk_and_blob_mapping
from aidb.utils.logger import logger
from tests.inference_service_utils.http_inference_service_setup import \
    run_server
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.tasti_test.tasti_test import TastiTests, VectorDatabaseType
from tests.utils import setup_gt_and_aidb_engine, setup_test_logger

setup_test_logger('limit_engine')

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

class LimitEngineTests(IsolatedAsyncioTestCase):

  def _infer_call_count(self, engine):
    return engine._config.inference_services['objects00'].infer_one.calls
  
  async def test_jackson_number_objects(self):
    '''
    Check the correctness of the limit engine results and the usage of the proxy score.
    Do query using random and accurate proxy scores and compare the infer count to validate proxy usage.
    The accurate proxy scores are generated based on the ground truth results.
    '''
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    p = Process(target=run_server, args=[str(data_dir)])
    p.start()
    time.sleep(1)

    # vector database configuration
    index_name = 'tasti'
    data_size = 1000
    embedding_dim = 128
    nb_buckets = 100
    vector_database_type = VectorDatabaseType.FAISS.value

    tasti_test = TastiTests(index_name, data_size, embedding_dim, nb_buckets, vector_database_type, index_path='./')
    tasti_index = tasti_test.tasti

    queries = [
      (
        '''SELECT * FROM colors02 WHERE frame >= 1000 and colors02.color = 'black' LIMIT 100;''',
        '''SELECT * FROM colors02 WHERE frame >= 1000 and colors02.color = 'black';''',
        '''SELECT {blob_mapping_table}.__vector_id FROM colors02 INNER JOIN {blob_mapping_table}
        ON colors02.frame = {blob_mapping_table}.frame WHERE colors02.frame >= 1000 AND colors02.color = 'black';'''
      ),
      (
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2 = 'green' LIMIT 100;''',
        '''SELECT frame, light_1, light_2 FROM lights01 WHERE light_2 = 'green';''',
        '''SELECT {blob_mapping_table}.__vector_id FROM lights01 INNER JOIN {blob_mapping_table}
        ON lights01.frame = {blob_mapping_table}.frame WHERE lights01.light_2 = 'green';'''
      ),
      (
        '''SELECT * FROM objects00 WHERE object_name = 'car' OR frame < 100 LIMIT 100;''',
        '''SELECT * FROM objects00 WHERE object_name = 'car' OR frame < 100;''',
        '''SELECT {blob_mapping_table}.__vector_id FROM objects00 INNER JOIN {blob_mapping_table}
        ON objects00.frame = {blob_mapping_table}.frame WHERE objects00.object_name = 'car' OR objects00.frame < 100;'''
      )
    ]
    db_url_list = [MYSQL_URL, SQLITE_URL, POSTGRESQL_URL]
    for db_url in db_url_list:
      dialect = db_url.split('+')[0]
      logger.info(f'Test {dialect} database')
      for aidb_query, exact_query, get_vector_id_query in queries:
        gt_engine, aidb_engine = await setup_gt_and_aidb_engine(db_url, data_dir, tasti_index)
        register_inference_services(aidb_engine, data_dir, batch_supported = False)
        logger.info(f'Running query {exact_query} in ground truth database')
        try:
          async with gt_engine.begin() as conn:
            gt_res = await conn.execute(text(exact_query))
            gt_res = gt_res.fetchall()
        finally:
          await gt_engine.dispose()

        # TODO: add cache clearing
        # Run query with random proxy score
        logger.info(f'Running query {aidb_query} in aidb database using random proxy score')
        infer_count_before_query = self._infer_call_count(aidb_engine)
        rand_proxy_aidb_res = aidb_engine.execute(aidb_query)
        rand_proxy_infer_count = self._infer_call_count(aidb_engine) - infer_count_before_query

        # TODO: add cache clearing
        # Run query with accurate proxy score
        logger.info(f'Running query {aidb_query} in aidb database using accurate proxy score')
        # Generate proxy score using the ground truth data
        _, _, blob_mapping_table_name = table_name_for_rep_and_topk_and_blob_mapping(['blobs_00'])
        try:
          async with gt_engine.begin() as conn:
            gt_res_vector_id = await conn.execute(text(get_vector_id_query.format(blob_mapping_table = 
                                                                                  blob_mapping_table_name)))
            gt_res_vector_id = set(gt_res_vector_id.fetchall())
        finally:
          await gt_engine.dispose()
        accurate_proxy_score = [(1 if (vector_id,) in gt_res_vector_id else 0) for vector_id in range(data_size)]
        accurate_proxy_score = pd.Series(accurate_proxy_score, index = range(data_size))
        # Execute query with given proxy score
        infer_count_before_query = self._infer_call_count(aidb_engine)
        accurate_proxy_aidb_res = aidb_engine.execute(aidb_query, 
                                                      proxy_score_for_all_blobs = accurate_proxy_score)
        accurate_proxy_infer_count = self._infer_call_count(aidb_engine) - infer_count_before_query

        # Check result correctness
        logger.info(f'There are {len(rand_proxy_aidb_res)} elements in aidb result using random proxy, '
              f'{len(accurate_proxy_aidb_res)} elements in aidb result using accurate proxy '
              f'and {len(gt_res)} elements in ground truth results')
        
        if len(rand_proxy_aidb_res) < 100:
          assert len(rand_proxy_aidb_res) == len(gt_res)
        else:
          assert len(rand_proxy_aidb_res) == 100
        # TODO: check this using the new check
        assert len(set(rand_proxy_aidb_res) - set(gt_res)) == 0

        if len(accurate_proxy_aidb_res) < 100:
          assert len(accurate_proxy_aidb_res) == len(gt_res)
        else:
          assert len(accurate_proxy_aidb_res) == 100
        # TODO: check this using the new check
        assert len(set(accurate_proxy_aidb_res) - set(gt_res)) == 0

        # Check call count
        logger.info(f'Inference service was called {rand_proxy_infer_count} times when using random proxy '
                    f'and {accurate_proxy_infer_count} times when using accurate proxy')
        assert accurate_proxy_infer_count <= rand_proxy_infer_count

      del gt_engine
      del aidb_engine
    p.terminate()
    p.join()

if __name__ == '__main__':
  unittest.main()