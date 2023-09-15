import asyncio

from tests.db_utils import create_db, setup_db, insert_data_in_tables, clear_all_tables, setup_config_tables

DB_URL = "postgresql+asyncpg://postgres@localhost"


async def main():
  data_dir = '/home/akash/Documents/aidb-new/data/jackson'
  # Set up the ground truth database
  gt_db_fname = 'aidb_gt'
  await create_db(DB_URL, gt_db_fname)
  gt_engine = await setup_db(DB_URL, gt_db_fname, data_dir)
  await insert_data_in_tables(gt_engine, data_dir, False)
  del gt_engine

  # Set up the aidb database
  aidb_db_fname = 'aidb_test'
  await create_db(DB_URL, aidb_db_fname)
  tmp_engine = await setup_db(DB_URL, aidb_db_fname, data_dir)
  await clear_all_tables(tmp_engine)
  await insert_data_in_tables(tmp_engine, data_dir, True)
  await setup_config_tables(tmp_engine)
  del tmp_engine


if __name__ == '__main__':
  asyncio.run(main())
