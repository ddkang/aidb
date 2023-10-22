import os
import unittest

from unittest import IsolatedAsyncioTestCase
from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables
from tests.inference_service_utils.inference_service_setup import register_inference_services
from aidb.engine import Engine

# DB_URL = "postgresql+asyncpg://postgres@localhost"
DB_URL = "sqlite+aiosqlite://"


class DbSetupTests(IsolatedAsyncioTestCase):

  async def test_positive(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/amazon')

    # Set up the aidb database
    aidb_db_fname = 'aidb_test_amazon.sqlite'
    await create_db(DB_URL, aidb_db_fname)

    tmp_engine = await setup_db(DB_URL, aidb_db_fname, data_dir)
    await setup_config_tables(tmp_engine)
    del tmp_engine
    # Connect to the aidb database
    aidb_engine = Engine(
      f'{DB_URL}/{aidb_db_fname}',
      debug=False,
    )
    register_inference_services(aidb_engine, data_dir)
    del aidb_engine


if __name__ == '__main__':
  unittest.main()
