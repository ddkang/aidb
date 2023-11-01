import argparse

from aidb_utilities.command_line_setup.command_line_setup import command_line_utility
from aidb_utilities.aidb_setup.aidb_factory import AIDB


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)

  args = parser.parse_args()
  command_line_utility(AIDB.from_config(args.config))
