import argparse
import importlib

import pandas as pd

from aidb_utilities.command_line_setup.command_line_setup import command_line_utility
from aidb_utilities.aidb_setup.aidb_factory import AIDB
from aidb_utilities.db_setup.blob_table import BaseTablesSetup
from aidb_utilities.db_setup.create_tables import create_output_tables
from aidb.utils.asyncio import asyncio_run
from aidb_utilities.db_setup.clear_cache import clear_ML_cache

def setup_blob_tables(config):
  input_blobs = pd.read_csv(config.blobs_csv_file)
  base_table_setup = BaseTablesSetup(f"{config.DB_URL}/{config.DB_NAME}")
  base_table_setup.insert_blob_meta_data(config.blob_table_name, input_blobs, config.blobs_keys_columns)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)
  parser.add_argument("--setup-blob-table", action='store_true')
  parser.add_argument("--setup-output-tables", action='store_true')
  parser.add_argument("--verbose", action='store_true')
  parser.add_argument("--clear-cache", action='store_true')
  args = parser.parse_args()

  config = importlib.import_module(args.config)

  if args.setup_blob_table:
    setup_blob_tables(config)

  if args.setup_output_tables:
    asyncio_run(create_output_tables(config.DB_URL, config.DB_NAME, config.tables))

  aidb_engine = AIDB.from_config(args.config, args.verbose)
  
  if args.clear_cache:
    asyncio_run(clear_ML_cache(aidb_engine))
  
  command_line_utility(aidb_engine)
