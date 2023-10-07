import os
import unittest

from unittest import IsolatedAsyncioTestCase
from tests.db_utils.db_setup import create_db, setup_db, setup_config_tables
from aidb.engine import Engine
from aidb.query.query import Query

DB_URL = "sqlite+aiosqlite://"


class ColumnsInQueryTests(IsolatedAsyncioTestCase):

  async def test_positive_object_detection(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')

    # Set up the aidb database
    aidb_db_fname = 'aidb_test.sqlite'
    await create_db(DB_URL, aidb_db_fname)

    tmp_engine = await setup_db(DB_URL, aidb_db_fname, data_dir)
    await setup_config_tables(tmp_engine)
    del tmp_engine
    # Connect to the aidb database
    aidb_engine = Engine(
      f'{DB_URL}/{aidb_db_fname}',
      debug=False,
    )
    # pairs of query, number of columns
    test_query_list = [("SELECT * FROM objects00;", 8),
                       ("SELECT object_id FROM objects00 WHERE frame > 100;", 2),
                       ("SELECT * FROM lights01,counts03;", 7),
                       ("SELECT l.frame,c.count FROM lights01 l JOIN counts03 c ON l.frame=c.frame;", 3),
                       ]
    for query, ground_truth in test_query_list:
      q = Query(query, aidb_engine._config)
      columns_in_query = q.columns_in_query
      assert len(columns_in_query) == ground_truth
    del aidb_engine


if __name__ == '__main__':
  unittest.main()
