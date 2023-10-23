import time

import pandas as pd

from aidb.engine import Engine
from tests.db_utils.db_setup import (clear_all_tables, create_db,
                                     insert_data_in_tables,
                                     setup_config_tables, setup_db)


async def setup_aidb_engine(db_url, aidb_db_fname, data_dir):
  await create_db(db_url, aidb_db_fname)
  tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
  await clear_all_tables(tmp_engine)
  await insert_data_in_tables(tmp_engine, data_dir, True)
  await setup_config_tables(tmp_engine)
  del tmp_engine


async def setup_gt_and_aidb_engine(db_url, data_dir, tasti_index=None, blob_mapping_table_name=None):
  # Set up the ground truth database
  gt_db_fname = 'aidb_gt'
  await create_db(db_url, gt_db_fname)
  gt_engine = await setup_db(db_url, gt_db_fname, data_dir)
  await insert_data_in_tables(gt_engine, data_dir, False)

  # Set up the aidb database
  aidb_db_fname = 'aidb_test'
  await setup_aidb_engine(db_url, aidb_db_fname, data_dir)
  # Connect to the aidb database
  engine = Engine(
    f'{db_url}/{aidb_db_fname}',
    debug=False,
    blob_mapping_table_name=blob_mapping_table_name,
    tasti_index=tasti_index
  )
  return gt_engine, engine


def command_line_utility(engine: Engine):
  with open('assets/welcome.txt', 'r') as content_file:
    content = content_file.read()
    print(content)
  print("Query AIDB using SQL....\n")
  while True:
    query = input(">>>")
    if query.strip() == "exit":
      return
    else:
      while query.strip()[-1] != ';':
        q = input("")
        query += f" {q.strip()}"
      try:
        print("Running...")
        start_time = time.time()
        results = engine.execute(query)
        end_time = time.time()
        print(f"Query Execution Time = {int((end_time - start_time)*100)/100} seconds")
        print("Query Result", results[0][0])
      except Exception as e:
        print(e)
