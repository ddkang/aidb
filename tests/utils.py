import logging
from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables, insert_data_in_tables, clear_all_tables
from aidb.engine import Engine
from aidb.utils.logger import logger

async def setup_gt_and_aidb_engine(db_url, data_dir, tasti_index = None, port = 8000):
  # Set up the ground truth database
  dialect = db_url.split("+")[0]
  if dialect == "postgresql" or dialect == "mysql":
    gt_db_fname = f'aidb_gt_{port}'
    aidb_db_fname = f'aidb_test_{port}'
  elif dialect == "sqlite":
    gt_db_fname = f'aidb_gt_{port}.sqlite'
    aidb_db_fname = f'aidb_test_{port}.sqlite'
  else:
    raise Exception('Unsupported database. We support mysql, sqlite and postgresql currently.')

  await create_db(db_url, gt_db_fname)
  gt_engine = await setup_db(db_url, gt_db_fname, data_dir)
  try:
    async with gt_engine.begin() as conn:
      await insert_data_in_tables(conn, data_dir, False)
  finally:
    await gt_engine.dispose()

  # Set up the aidb database
  await create_db(db_url, aidb_db_fname)
  tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
  try:
    async with tmp_engine.begin() as conn:
      await clear_all_tables(conn)
      await insert_data_in_tables(conn, data_dir, True)
      await setup_config_tables(conn)
  finally:
    await tmp_engine.dispose()

  # Connect to the aidb database
  engine = Engine(
    f'{db_url}/{aidb_db_fname}',
    debug=False,
    tasti_index=tasti_index
  )

  return gt_engine, engine


def setup_test_logger(log_fname):
  logger.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
  
  file_handler = logging.FileHandler(f'{log_fname}.log')
  file_handler.setLevel(logging.INFO)
  file_handler.setFormatter(formatter)
  
  logger.addHandler(file_handler)