import asyncio

from tests.db_utils.db_setup import create_db, setup_db, insert_data_in_tables, clear_all_tables, setup_config_tables
from tests.inference_service_utils.inference_service_setup import register_inference_services
from aidb.engine import Engine
import unittest

DB_URL = "postgresql+asyncpg://postgres@localhost"


# DB_URL = "sqlite+aiosqlite://"

async def main():
  data_dir = '/tests/data/jackson'
  # Set up the aidb database
  aidb_db_fname = 'aidb_test'
  await create_db(DB_URL, aidb_db_fname)

  tmp_engine = await setup_db(DB_URL, aidb_db_fname, data_dir)
  await clear_all_tables(tmp_engine)
  await setup_config_tables(tmp_engine)
  del tmp_engine
  # Connect to the aidb database
  aidb_engine = Engine(
    f'{DB_URL}/{aidb_db_fname}',
    debug=False,
  )
  register_inference_services(aidb_engine, data_dir)


if __name__ == '__main__':
  asyncio.run(main())

# class GraphTests(unittest.TestCase):
#   async def test_1(self):
#     g = Graph({"a": ["b", "c"], "b": ["c"]})
#     self.assertTrue(g.isCyclic())
#
#   def test_2(self):
#     g = Graph({"a": ["b", "c"], "c": ["d"]})
#     self.assertFalse(g.isCyclic())
#
#   def test_3(self):
#     g = Graph({"a": ["b", "c"], "c": ["d"], "d": ["a"]})
#     self.assertTrue(g.isCyclic())
#
#
# if __name__ == '__main__':
#   unittest.main()
