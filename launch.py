import argparse
from typing import Dict, List

import pandas as pd
import yaml

from aidb.utils.asyncio import asyncio_run
from aidb_utilities.aidb_setup.aidb_factory import AIDB
from aidb_utilities.command_line_setup.command_line_setup import \
    command_line_utility
from aidb_utilities.db_setup.blob_table import BaseTablesSetup
from aidb_utilities.db_setup.create_tables import create_output_tables


def setup_blob_tables(db_config: Dict[str, str], blob_tables: List[dict]):
  base_table_setup = BaseTablesSetup(f"{db_config['url']}/{db_config['name']}")
  for blob_table in blob_tables:
    input_blobs = pd.read_csv(blob_table['path'])
    base_table_setup.insert_blob_meta_data(
      blob_table['name'], input_blobs, blob_table['keys'])


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)
  parser.add_argument("--setup-blob-table", action='store_true')
  parser.add_argument("--setup-output-tables", action='store_true')
  parser.add_argument("--verbose", action='store_true')

  args = parser.parse_args()
  with open(args.config, "r") as file:  # Replace with the actual path to your YAML file
    config = yaml.safe_load(file)

  if args.setup_blob_table:
    setup_blob_tables(config['db_config'], config['blob_tables'])

  if args.setup_output_tables:
    asyncio_run(create_output_tables(
      config['db_config'], config['output_tables']))

  aidb_engine = AIDB.from_config(config, args.verbose)
  command_line_utility(aidb_engine)
