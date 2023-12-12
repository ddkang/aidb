import os
import unittest

from collections import Counter
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.utils import setup_gt_and_aidb_engine

DB_URL = "sqlite+aiosqlite://"

class QueryParsingTests(IsolatedAsyncioTestCase):
  def are_lists_equal(self, list1, list2):
    for sub_list1, sub_list2 in zip(list1, list2):
      assert Counter(sub_list1) == Counter(sub_list2)

  def _test_query(self, query_str, config, normalized_query, correct_fp, correct_service, correct_tables, num_query):
    query = Query(query_str, config)
    # test the number of queries
    assert len(query.all_queries_in_expressions) == num_query
    self.assertEqual(query.query_after_normalizing.sql_str, normalized_query)
    and_fp = []
    for and_connected in query.filtering_predicates:
      or_fp = []
      for or_connected in and_connected:
        or_fp.append(or_connected.sql())
      and_fp.append(or_fp)
    self.are_lists_equal(and_fp, correct_fp)
    self.are_lists_equal(query.inference_engines_required_for_filtering_predicates, correct_service)
    self.are_lists_equal(query.tables_in_filtering_predicates, correct_tables)


  async def test_nested_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config

    # test column alias, subquery
    query_str1 = '''
                 SELECT color as alias_color 
                 FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame 
                 WHERE alias_color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2) 
                  OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100);
                 '''

    normalized_query_str1 = ("SELECT colors02.color "
                             "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                             "WHERE colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2) "
                             "OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)")

    # test replacing column with root column in filtering predicate,
    # the root column of 'colors02.object_id' is 'objects00'
    correct_fp = [['colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2)',
                    'objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)']]

    # filter predicates connected by OR are in same set
    correct_service = [{'colors02', 'objects00'}]
    correct_tables = [{'colors02', 'objects00'}]
    self._test_query(query_str1, config, normalized_query_str1, correct_fp, correct_service, correct_tables, 3)


    # test table alias
    query_str2 = '''
                 SELECT table1.color FROM colors02 AS table1 WHERE frame IN (SELECT * FROM blobs_00)
                 AND object_id > 0
                 '''

    normalized_query_str2 = ("SELECT colors02.color FROM colors02 "
                             "WHERE colors02.frame IN (SELECT * FROM blobs_00) "
                             "AND colors02.object_id > 0")

    # test replacing column with root column in filtering predicate,
    # the root column of 'colors02.object_id' is 'objects00'
    correct_fp = [['blobs_00.frame IN (SELECT * FROM blobs_00)'],
                   ['objects00.object_id > 0']]
    # filter predicates connected by AND are in different set
    correct_service = [set(), {'objects00'}]
    correct_tables = [set(), {'objects00'}]
    self._test_query(query_str2, config, normalized_query_str2, correct_fp, correct_service, correct_tables, 2)


    # test sub-subquery
    query_str3 = '''SELECT frame, object_id FROM colors02 AS cl 
                    WHERE cl.object_id > (SELECT AVG(object_id) FROM objects00 
                                          WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))
                 '''
    normalized_query_str3 = ("SELECT colors02.frame, colors02.object_id FROM colors02 "
                             "WHERE colors02.object_id > (SELECT AVG(object_id) FROM objects00 WHERE frame > "
                             "(SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))")
    correct_fp = [["objects00.object_id > (SELECT AVG(object_id) FROM objects00 "
                   "WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))"]]

    correct_service = [{'objects00'}]
    correct_tables = [{'objects00'}]
    self._test_query(query_str3, config, normalized_query_str3, correct_fp, correct_service, correct_tables, 3)


    # test multiple aliases
    query_str4 = '''
                 SELECT color, table2.x_min
                 FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame 
                 WHERE color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3) 
                  AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 '''

    normalized_query_str4 = ("SELECT colors02.color, objects00.x_min "
                             "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                             "WHERE colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
                             "AND objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)")

    correct_fp = [["colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3)"],
                   ["objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]]

    correct_service = [{'colors02'}, {'objects00'}]
    correct_tables = [{'colors02'}, {'objects00'}]
    self._test_query(query_str4, config, normalized_query_str4, correct_fp, correct_service, correct_tables, 3)


    # comparison between subquery
    query_str5 = '''
                 SELECT color, table2.x_min
                 FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame 
                 WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3) 
                    > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 '''

    normalized_query_str5 = ("SELECT colors02.color, objects00.x_min "
                             "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                             "WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
                             "> (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)")

    correct_fp = [["(SELECT table3.color AS alias_color2 FROM colors02 AS table3) > "
                   "(SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]]

    correct_service = [{}]
    correct_tables = [{}]
    self._test_query(query_str5, config, normalized_query_str5, correct_fp, correct_service, correct_tables, 3)


    query_str6 = '''
                 SELECT color AS col1, table2.x_min AS col2, table2.y_min AS col3
                 FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame;
                 '''

    normalized_query_str6 = ("SELECT colors02.color, objects00.x_min, objects00.y_min "
                             "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame")

    correct_fp = [[]]

    correct_service = [{}]
    correct_tables = [{}]
    self._test_query(query_str6, config, normalized_query_str6, correct_fp, correct_service, correct_tables, 1)


    query_str7 = '''
                 SELECT color AS col1, table2.x_min AS col2, table3.frame AS col3
                 FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
                 JOIN blobs_00 table3 ON table2.frame = table3.frame;
                 '''

    normalized_query_str7 = ("SELECT colors02.color, objects00.x_min, blobs_00.frame "
                             "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                             "JOIN blobs_00 ON objects00.frame = blobs_00.frame")

    correct_fp = [[]]

    correct_service = [{}]
    correct_tables = [{}]
    self._test_query(query_str7, config, normalized_query_str7, correct_fp, correct_service, correct_tables, 1)


    query_str8 = '''
                 SELECT color, x_min AS col2, colors02.frame AS col3
                 FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
                 WHERE color = 'blue' AND x_min > 600;
                 '''

    normalized_query_str8 = ("SELECT colors02.color, objects00.x_min, colors02.frame "
                             "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
                             "WHERE colors02.color = 'blue' AND objects00.x_min > 600")

    correct_fp = [["colors02.color = 'blue'"], ['objects00.x_min > 600']]

    correct_service = [{'colors02'}, {'objects00'}]
    correct_tables = [{'colors02'}, {'objects00'}]
    self._test_query(query_str8, config, normalized_query_str8, correct_fp, correct_service, correct_tables, 1)


    query_str9 = '''
                 SELECT color, userfunction(x_min, y_min, x_max, y_max)
                 FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
                 WHERE table2.frame > 10000 OR y_max < 800;
                 '''

    normalized_query_str9 = ("SELECT colors02.color, userfunction(objects00.x_min, objects00.y_min, "
                             "objects00.x_max, objects00.y_max) "
                             "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
                             "WHERE objects00.frame > 10000 OR objects00.y_max < 800")

    correct_fp = [['blobs_00.frame > 10000', 'objects00.y_max < 800']]

    correct_service = [{'objects00'}]
    correct_tables = [{'objects00'}]
    self._test_query(query_str9, config, normalized_query_str9, correct_fp, correct_service, correct_tables, 1)


  # We don't support using approximate query as a subquery
  async def test_approximate_query_as_subquery(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config

    # approx aggregation as a subquery
    query_str1 = '''
                 SELECT x_min FROM  objects00
                 WHERE x_min > (SELECT AVG(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%) 
                 AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 '''

    parsed_query = Query(query_str1, config)
    with self.assertRaises(Exception):
      _ = parsed_query.all_queries_in_expressions

    # approx select as a subquery
    query_str2 = '''
                 SELECT x_min FROM  objects00
                 WHERE frame IN (SELECT frame FROM colors02 where color LIKE 'blue' 
                    RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;) 
                 AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 '''

    parsed_query = Query(query_str2, config)
    with self.assertRaises(Exception):
      _ = parsed_query.all_queries_in_expressions


  async def test_correct_approximate_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config

    # approx aggregation
    query_str1 = '''
                 SELECT AVG(x_min) FROM  objects00
                 WHERE frame > (SELECT AVG(frame) FROM blobs_00) 
                 ERROR_TARGET 10% CONFIDENCE 95%;
                 '''

    normalized_query_str1 = ("SELECT AVG(objects00.x_min) "
                             "FROM objects00 "
                             "WHERE objects00.frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%")

    correct_fp = [["blobs_00.frame > (SELECT AVG(frame) FROM blobs_00)"]]

    correct_service = [{}]
    correct_tables = [{}]
    self._test_query(query_str1, config, normalized_query_str1, correct_fp, correct_service, correct_tables, 2)


    # approx select as a subquery
    query_str2 = '''
                 SELECT frame FROM colors02 
                 WHERE color IN (SELECT color FROM colors02 WHERE frame > 10000)
                 RECALL_TARGET 80% 
                 CONFIDENCE 95%;
                 '''

    normalized_query_str2 = ("SELECT colors02.frame FROM colors02 "
                             "WHERE colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000) "
                             "RECALL_TARGET 80% "
                             "CONFIDENCE 95%")

    correct_fp = [["colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000)"]]

    correct_service = [{'colors02'}]
    correct_tables = [{'colors02'}]
    self._test_query(query_str2, config, normalized_query_str2, correct_fp, correct_service, correct_tables, 2)


if __name__ == '__main__':
  unittest.main()
