from multiprocessing import Process
import os
import unittest
from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text


from tests.db_utils.db_setup import (create_db, setup_db, setup_config_tables,
                                    insert_data_in_tables, clear_all_tables)
from tests.inference_service_utils.inference_service_setup import register_inference_services
from tests.inference_service_utils.http_inference_service_setup import run_server


from aidb.engine.engine import Engine


DB_URL = "sqlite+aiosqlite://"
queries = [
      (
        'aggregate',
        '''SELECT AVG(x_min) FROM objects00;''',
        '''SELECT AVG(x_min) FROM objects00;''',
      )
    ]

class AggeregateEngineTests(IsolatedAsyncioTestCase):
  async def test_agg_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    p = Process(target=run_server, args=[str(data_dir)])
    p.start()

    # Set up the ground truth database
    gt_db_fname = 'aidb_gt'
    await create_db(DB_URL, gt_db_fname)
    gt_engine = await setup_db(DB_URL, gt_db_fname, data_dir)
    await insert_data_in_tables(gt_engine, data_dir, False)

    # Set up the aidb database
    aidb_db_fname = 'aidb_test'
    await create_db(DB_URL, aidb_db_fname)
    tmp_engine = await setup_db(DB_URL, aidb_db_fname, data_dir)
    await clear_all_tables(tmp_engine)
    await insert_data_in_tables(tmp_engine, data_dir, True)
    await setup_config_tables(tmp_engine)
    del tmp_engine

    # Connect to the aidb database
    engine = Engine(
      f'{DB_URL}/{aidb_db_fname}',
      debug=False,
    )
    register_inference_services(engine, data_dir, 'objects00')

    for query_type, aidb_query, aggregate_query in queries:

      print(f'Running query {aggregate_query} in ground truth database')
      async with gt_engine.begin() as conn:
        gt_res = await conn.execute(text(aggregate_query))
        gt_res = gt_res.fetchall()

      print(f'Running query {aidb_query} in aidb database')
      aidb_res = engine.execute(aidb_query)

      print(aidb_res, gt_res, 'aidb_res, gt_res')

      # TODO: equality check should be implemented
      assert len(gt_res) == len(aidb_res)

      del gt_engine
      p.terminate()


if __name__ == '__main__':
  unittest.main()
