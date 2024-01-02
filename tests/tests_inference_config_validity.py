import os
import unittest

from unittest import IsolatedAsyncioTestCase
from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables
from tests.inference_service_utils.inference_service_setup import register_inference_services
from aidb.engine import Engine


class InferenceConfigIntegrityTests(IsolatedAsyncioTestCase):
  async def _test_positive_object_detection(self, db_url, aidb_db_fname):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    # Set up the aidb database
    await create_db(db_url, aidb_db_fname)

    tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)

    try:
      async with tmp_engine.begin() as conn:
        await setup_config_tables(conn)
    except Exception:
      raise Exception('Fail to setup config table.')
    finally:
      await tmp_engine.dispose()

    del tmp_engine

    # Connect to the aidb database
    aidb_engine = Engine(
      f'{db_url}/{aidb_db_fname}',
      debug=False,
    )
    register_inference_services(aidb_engine, data_dir)
    del aidb_engine


  async def _test_positive_only_1_table(self, db_url, aidb_db_fname):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/twitter')

    # Set up the aidb database
    await create_db(db_url, aidb_db_fname)

    tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)

    try:
      async with tmp_engine.begin() as conn:
        await setup_config_tables(conn)
    except Exception:
      raise Exception('Fail to setup config table.')
    finally:
      await tmp_engine.dispose()

    del tmp_engine
    # Connect to the aidb database
    aidb_engine = Engine(
      f'{db_url}/{aidb_db_fname}',
      debug=False,
    )
    register_inference_services(aidb_engine, data_dir)
    del aidb_engine


  async def _test_negative_col_by_multiple(self, db_url, aidb_db_fname):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/same_col_by_multiple_services')

    # Set up the aidb database
    await create_db(db_url, aidb_db_fname)

    tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
    try:
      async with tmp_engine.begin() as conn:
        await setup_config_tables(conn)
    except Exception:
      raise Exception('Fail to setup config table.')
    finally:
      await tmp_engine.dispose()

    del tmp_engine
    # Connect to the aidb database
    aidb_engine = Engine(
      f'{db_url}/{aidb_db_fname}',
      debug=False,
    )
    with self.assertRaises(Exception):
      register_inference_services(aidb_engine, data_dir)
    del aidb_engine


  async def _test_positive_multi_table_input(self, db_url, aidb_db_fname):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/multi_table_input')

    # Set up the aidb database
    await create_db(db_url, aidb_db_fname)

    tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
    try:
      async with tmp_engine.begin() as conn:
        await setup_config_tables(conn)
    except Exception:
      raise Exception('Fail to setup config table.')
    finally:
      await tmp_engine.dispose()

    del tmp_engine
    # Connect to the aidb database
    aidb_engine = Engine(
      f'{db_url}/{aidb_db_fname}',
      debug=False,
    )
    register_inference_services(aidb_engine, data_dir)
    del aidb_engine


  async def test_all_tests(self):
    db_configs = [
      {'db_url': 'sqlite+aiosqlite://', 'db_name': 'aidb_test.sqlite'},
      {'db_url': 'postgresql+asyncpg://user:testaidb@localhost:5432', 'db_name': 'aidb_test'},
      {'db_url': 'mysql+aiomysql://root:testaidb@localhost:3306', 'db_name': 'aidb_test'}
    ]

    for config in db_configs:
      await self._test_positive_object_detection(config['db_url'], config['db_name'])
      await self._test_positive_only_1_table(config['db_url'], config['db_name'])
      await self._test_positive_multi_table_input(config['db_url'], config['db_name'])
      await self._test_negative_col_by_multiple(config['db_url'], config['db_name'])


if __name__ == '__main__':
  unittest.main()
