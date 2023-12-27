import os
import unittest

from unittest import IsolatedAsyncioTestCase
from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables
from tests.inference_service_utils.inference_service_setup import register_inference_services
from aidb.engine import Engine

POSTGRESQL_URL = 'postgresql+asyncpg://user:testaidb@localhost:5432'
SQLITE_URL = 'sqlite+aiosqlite://'
MYSQL_URL = 'mysql+aiomysql://root:testaidb@localhost:3306'

class DbSetupTests(IsolatedAsyncioTestCase):
  async def test_positive(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    db_url_list = [SQLITE_URL, POSTGRESQL_URL, MYSQL_URL]
    for db_url in db_url_list:
      # Set up the aidb database
      if db_url == SQLITE_URL:
        aidb_db_fname = 'aidb_test.sqlite'
      else:
        aidb_db_fname = 'aidb_test'

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


if __name__ == '__main__':
  unittest.main()
