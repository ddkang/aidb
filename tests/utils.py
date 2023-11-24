from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables, insert_data_in_tables, clear_all_tables
from aidb.engine import Engine

async def setup_gt_and_aidb_engine(db_url, data_dir, tasti_index = None, port = 8000):
  # Set up the ground truth database
  gt_db_fname = f'aidb_gt_{port}.sqlite'
  await create_db(db_url, gt_db_fname)
  gt_engine = await setup_db(db_url, gt_db_fname, data_dir)
  await insert_data_in_tables(gt_engine, data_dir, False)

  # Set up the aidb database
  aidb_db_fname = f'aidb_test_{port}.sqlite'
  await create_db(db_url, aidb_db_fname)
  tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
  await clear_all_tables(tmp_engine)
  await insert_data_in_tables(tmp_engine, data_dir, True)
  await setup_config_tables(tmp_engine)
  del tmp_engine
  # Connect to the aidb database
  engine = Engine(
    f'{db_url}/{aidb_db_fname}',
    debug=False,
    tasti_index=tasti_index
  )

  return gt_engine, engine