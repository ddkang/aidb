import time

from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables, insert_data_in_tables, clear_all_tables


# TODO: replace this with AIDB utilities
async def setup_aidb_engine(db_url, aidb_db_fname, data_dir):
  await create_db(db_url, aidb_db_fname)
  tmp_engine = await setup_db(db_url, aidb_db_fname, data_dir)
  await clear_all_tables(tmp_engine)
  await insert_data_in_tables(tmp_engine, data_dir, True)
  await setup_config_tables(tmp_engine)
  del tmp_engine


def command_line_utility(engine):
  welcome = '''
      _    ___ ____  ____  
     / \  |_ _|  _ \| __ ) 
    / _ \  | || | | |  _ \ 
   / ___ \ | || |_| | |_) |
  /_/   \_\___|____/|____/ 
                           
  '''
  print(welcome)
  print("Query AIDB using SQL....\n")
  while True:
    query = input(">>>")
    if query.strip() == "exit":
      return
    else:
      while query.strip()[-1] != ';':
        q = input("")
        query += f" {q.strip()}"
      print("Running...")
      start_time = time.time()
      results = engine.execute(query)
      end_time = time.time()
      print(f"Query Execution Time = {int((end_time - start_time)*100)/100} seconds")
      print("Query Result", results)
