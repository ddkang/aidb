import argparse

from tests.utils import command_line_utility
from aidb_factory import AIDB


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str)

  args = parser.parse_args()
  command_line_utility(AIDB.from_config(args.config))
