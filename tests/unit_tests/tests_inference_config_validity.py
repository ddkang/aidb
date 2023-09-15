import asyncio

from tests.db_utils.db_setup import create_db, setup_db, insert_data_in_tables, clear_all_tables, setup_config_tables
from tests.inference_service_utils import register_inference_services
from aidb.engine import Engine
from sqlalchemy.sql import text
import unittest


DB_URL = "postgresql+asyncpg://postgres@localhost"
# DB_URL = "sqlite+aiosqlite://"

async def main():
  data_dir = '/home/akash/Documents/aidb-new/data/jackson'

  queries = [
    (
      'full_scan',
      '''SELECT * FROM objects00;''',
      '''SELECT * FROM objects00;''',
    )
  ]

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

  register_inference_services(engine, data_dir)

  for query_type, aidb_query, exact_query in queries:
    print(f'Running query {exact_query} in ground truth database')
    # Run the query on the ground truth database
    async with gt_engine.begin() as conn:
      gt_res = await conn.execute(text(exact_query))
      gt_res = gt_res.fetchall()
      print(gt_res)
    # Run the query on the aidb database
    print(f'Running query {aidb_query} in aidb database')
    aidb_res = engine.execute(aidb_query)
    # TODO: check that the results are the same
    print(gt_res[0])
    print(aidb_res[0])
    del gt_engine


# if __name__ == '__main__':
#   asyncio.run(main())




class GraphTests(unittest.TestCase):
  async def test_1(self):
    g = Graph({"a": ["b", "c"], "b": ["c"]})
    self.assertTrue(g.isCyclic())

  def test_2(self):
    g = Graph({"a": ["b", "c"], "c": ["d"]})
    self.assertFalse(g.isCyclic())

  def test_3(self):
    g = Graph({"a": ["b", "c"], "c": ["d"], "d": ["a"]})
    self.assertTrue(g.isCyclic())


if __name__ == '__main__':
  unittest.main()
