import argparse
import importlib

import pandas as pd

from aidb_utilities.command_line_setup.command_line_setup import command_line_utility
from aidb_utilities.aidb_setup.aidb_factory import AIDB
from aidb_utilities.db_setup.blob_table import BaseTablesSetup


def setup_blob_tables(config_path):
  config = importlib.import_module(config_path)
  input_blobs = pd.read_csv(config.blobs_csv_file)
  base_table_setup = BaseTablesSetup(f"{config.DB_URL}/{config.DB_NAME}")
  base_table_setup.insert_blob_meta_data(config.blob_table_name, input_blobs, config.blobs_keys_columns)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)
  parser.add_argument("--setup-blob-table", action='store_true')
  args = parser.parse_args()

  if args.setup_blob_table:
    setup_blob_tables(args.config)

  aidb_engine = AIDB.from_config(args.config)
  command_line_utility(aidb_engine)
