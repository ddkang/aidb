import os
import unittest

from collections import Counter
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.utils import setup_gt_and_aidb_engine

DB_URL = "sqlite+aiosqlite://"

# normal query extraction parameters
QUERY_STR = 'query_str'
NORMALIZED_QUERY_STR = 'normalized_query_str'
CORRECT_FP = 'correct_fp'
CORRECT_SERVICE = 'correct_service'
CORRECT_TABLES = 'correct_tables'
NUM_QUERY = 'num_query'

# udf query extraction parameters
QUERY_EXTRACTED = 'query_after_extraction'
DATAFRAME_SQL = 'dataframe_sql'
UDF_MAPPING = 'udf_mapping'
COL_NAMES = 'col_names'
FUNCTION_NAME = 'function_name'
RESULT_COL_NAME = 'result_col_name'
SELECT_COL = 'select_col'
FILTER_PREDICATE = 'filter_predicate'

class QueryParsingTests(IsolatedAsyncioTestCase):
  def are_lists_equal(self, list1, list2):
    for sub_list1, sub_list2 in zip(list1, list2):
      assert Counter(sub_list1) == Counter(sub_list2)

  def _test_query(self, test_query, config):
    query = Query(test_query[QUERY_STR], config)
    # test the number of queries
    assert len(query.all_queries_in_expressions) == test_query[NUM_QUERY]
    self.assertEqual(query.query_after_normalizing.sql_str, test_query[NORMALIZED_QUERY_STR])
    and_fp = []
    for and_connected in query.filtering_predicates:
      or_fp = []
      for or_connected in and_connected:
        or_fp.append(or_connected.sql())
      and_fp.append(or_fp)
    self.are_lists_equal(and_fp, test_query[CORRECT_FP])
    self.are_lists_equal(query.inference_engines_required_for_filtering_predicates, test_query[CORRECT_SERVICE])
    self.are_lists_equal(query.tables_in_filtering_predicates, test_query[CORRECT_TABLES])


  async def test_nested_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config
    
       
    queries = {
    "test_query_0": {
        QUERY_STR: '''
                    SELECT color as alias_color
                    FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame
                    WHERE alias_color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2)
                      OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100);
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.color "
                                "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                                "WHERE colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2) "
                                  "OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)"),
        # test replacing column with root column in filtering predicate,
        # the root column of 'colors02.object_id' is 'objects00'
        CORRECT_FP: [['colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2)',
                    'objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)']],
        # filter predicates connected by OR are in same set
        CORRECT_SERVICE: [{'colors02', 'objects00'}],
        CORRECT_TABLES: [{'colors02', 'objects00'}],
        NUM_QUERY: 3
    },
    # test table alias
    "test_query_1": {
        QUERY_STR: '''
                    SELECT table1.color FROM colors02 AS table1 WHERE frame IN (SELECT * FROM blobs_00)
                    AND object_id > 0
                    ''',
        # test table alias
        NORMALIZED_QUERY_STR: ("SELECT colors02.color FROM colors02 "
                                "WHERE colors02.frame IN (SELECT * FROM blobs_00) "
                                "AND colors02.object_id > 0"),
        # test replacing column with root column in filtering predicate,
        # the root column of 'colors02.object_id' is 'objects00'
        CORRECT_FP: [['blobs_00.frame IN (SELECT * FROM blobs_00)'],
                   ['objects00.object_id > 0']],
        # filter predicates connected by AND are in different set
        CORRECT_SERVICE: [set(), {'objects00'}],
        CORRECT_TABLES: [set(), {'objects00'}],
        NUM_QUERY: 2
    },
    # test sub-subquery
     "test_query_2": {
        QUERY_STR: '''SELECT frame, object_id FROM colors02 AS cl
                        WHERE cl.object_id > (SELECT AVG(object_id) FROM objects00
                                              WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.frame, colors02.object_id FROM colors02 "
                                "WHERE colors02.object_id > (SELECT AVG(object_id) FROM objects00 WHERE frame > "
                                "(SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))"),
        # test replacing column with root column in filtering predicate,
        # the root column of 'colors02.object_id' is 'objects00'
        CORRECT_FP: [["objects00.object_id > (SELECT AVG(object_id) FROM objects00 "
                   "WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))"]],
        # filter predicates connected by AND are in different set
        CORRECT_SERVICE: [{'objects00'}],
        CORRECT_TABLES:  [{'objects00'}],
        NUM_QUERY: 3
    },
    # test multiple aliases
     "test_query_3": {
        QUERY_STR: '''
                    SELECT color, table2.x_min
                    FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
                    WHERE color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3)
                      AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.color, objects00.x_min "
                                "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                                "WHERE colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
                                "AND objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"),
        CORRECT_FP: [["colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3)"],
                   ["objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]],
        CORRECT_SERVICE: [{'colors02'}, {'objects00'}],
        CORRECT_TABLES: [{'colors02'}, {'objects00'}],
        NUM_QUERY: 3
    },
     # comparison between subquery
     "test_query_4":{
        QUERY_STR: '''
                    SELECT color, table2.x_min
                    FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
                    WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3)
                        > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                    ''',
        # test table alias
        NORMALIZED_QUERY_STR: ("SELECT colors02.color, objects00.x_min "
                                "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                                "WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
                                "> (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"),
        CORRECT_FP: [["(SELECT table3.color AS alias_color2 FROM colors02 AS table3) > "
                   "(SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]],
      
        CORRECT_SERVICE: [{}],
        CORRECT_TABLES:[{}],
        NUM_QUERY: 3
    },
     "test_query_5":{
        QUERY_STR:'''
                    SELECT color AS col1, table2.x_min AS col2, table2.y_min AS col3
                    FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame;
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.color, objects00.x_min, objects00.y_min "
                                "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame"),
        CORRECT_FP: [[]],
        CORRECT_SERVICE: [{}],
        CORRECT_TABLES: [{}],
        NUM_QUERY: 1
    },
     "test_query_6":{
        QUERY_STR: '''
                    SELECT color AS col1, table2.x_min AS col2, table3.frame AS col3
                    FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
                    JOIN blobs_00 table3 ON table2.frame = table3.frame;
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.color, objects00.x_min, blobs_00.frame "
                                "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
                                "JOIN blobs_00 ON objects00.frame = blobs_00.frame"),
        CORRECT_FP: [[]],
        CORRECT_SERVICE: [{}],
        CORRECT_TABLES: [{}],
        NUM_QUERY: 1
    },
     "test_query_7":{
        QUERY_STR: '''
                    SELECT color, x_min AS col2, colors02.frame AS col3
                    FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
                    WHERE color = 'blue' AND x_min > 600;
                    ''',
        NORMALIZED_QUERY_STR:  ("SELECT colors02.color, objects00.x_min, colors02.frame "
                                "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
                                "WHERE colors02.color = 'blue' AND objects00.x_min > 600"),
        CORRECT_FP: [["colors02.color = 'blue'"], ['objects00.x_min > 600']],
        CORRECT_SERVICE: [{'colors02'}, {'objects00'}],
        CORRECT_TABLES: [{'colors02'}, {'objects00'}],
        NUM_QUERY: 1
    },
     "test_query_8":{
        QUERY_STR:'''
                    SELECT color, userfunction(x_min, y_min, x_max, y_max)
                    FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
                    WHERE table2.frame > 10000 OR y_max < 800;
                    ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.color, userfunction(objects00.x_min, objects00.y_min, "
                                "objects00.x_max, objects00.y_max) "
                                "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
                                "WHERE objects00.frame > 10000 OR objects00.y_max < 800"),
        CORRECT_FP: [['blobs_00.frame > 10000', 'objects00.y_max < 800']],
        CORRECT_SERVICE: [{'objects00'}],
        CORRECT_TABLES: [{'objects00'}],
        NUM_QUERY: 1
    }
}
    for i in range(len(queries)):
      self._test_query(queries[f'test_query_{i}'], config)


  # We don't support using approximate query as a subquery
  async def test_approximate_query_as_subquery(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config
    queries = {
      # approx aggregation as a subquery
      "query_str1": '''
                 SELECT x_min FROM  objects00
                 WHERE x_min > (SELECT AVG(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%)
                 AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 ''',
      # approx select as a subquery
      "query_str2": '''
                 SELECT x_min FROM  objects00
                 WHERE frame IN (SELECT frame FROM colors02 where color LIKE 'blue'
                    RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;)
                 AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
                 '''
    }
    parsed_query = Query(queries["query_str1"], config)
    with self.assertRaises(Exception):
      _ = parsed_query.all_queries_in_expressions

    parsed_query = Query(queries["query_str2"], config)
    with self.assertRaises(Exception):
      _ = parsed_query.all_queries_in_expressions


  async def test_correct_approximate_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config
    
    queries = {
    # approx aggregation as a subquery
    "test_query_0": {
        QUERY_STR: '''
                 SELECT AVG(x_min) FROM  objects00
                 WHERE frame > (SELECT AVG(frame) FROM blobs_00)
                 ERROR_TARGET 10% CONFIDENCE 95%;
                 ''',
        NORMALIZED_QUERY_STR: ("SELECT AVG(objects00.x_min) "
                             "FROM objects00 "
                             "WHERE objects00.frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%"),
 
        CORRECT_FP : [["blobs_00.frame > (SELECT AVG(frame) FROM blobs_00)"]],
        
        # filter predicates connected by OR are in same set
        CORRECT_SERVICE: [{}],
        CORRECT_TABLES: [{}],
        NUM_QUERY: 2
    },
     "test_query_1":{
        QUERY_STR: '''
                 SELECT frame FROM colors02
                 WHERE color IN (SELECT color FROM colors02 WHERE frame > 10000)
                 RECALL_TARGET 80%
                 CONFIDENCE 95%;
                 ''',
        NORMALIZED_QUERY_STR: ("SELECT colors02.frame FROM colors02 "
                             "WHERE colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000) "
                             "RECALL_TARGET 80% "
                             "CONFIDENCE 95%"),
        CORRECT_FP:[["colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000)"]],
        CORRECT_SERVICE: [{'colors02'}],
        CORRECT_TABLES: [{'colors02'}],
        NUM_QUERY: 2
    }
}
    for i in range(len(queries)):
      self._test_query(queries[f'test_query_{i}'], config)

  async def test_udf_query(self):
    def _test_equality(test_query, config):
      query = Query(test_query[QUERY_STR], config)
      dataframe_sql, query_after_extraction = query.udf_query
      self.assertEqual(query_after_extraction.sql_str, test_query[QUERY_EXTRACTED])
      assert len(dataframe_sql[UDF_MAPPING]) == len(test_query[DATAFRAME_SQL][UDF_MAPPING])
      assert all(any(e1 == e2 for e2 in dataframe_sql[UDF_MAPPING])
                for e1 in test_query[DATAFRAME_SQL][UDF_MAPPING])
      assert dataframe_sql[SELECT_COL] == test_query[DATAFRAME_SQL][SELECT_COL]
      filter_predicate = query.convert_and_connected_fp_to_exp(dataframe_sql[FILTER_PREDICATE])
      if filter_predicate:
        filter_predicate = filter_predicate.sql()

      assert filter_predicate == test_query[DATAFRAME_SQL][FILTER_PREDICATE]


    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    aidb_engine.register_user_defined_function('sum_function', None)
    aidb_engine.register_user_defined_function('is_equal', None)
    aidb_engine.register_user_defined_function('power_function', None)
    aidb_engine.register_user_defined_function('max_function', None)
    aidb_engine.register_user_defined_function('multiply_function', None)
    aidb_engine.register_user_defined_function('function1', None)
    aidb_engine.register_user_defined_function('function2', None)
    aidb_engine.register_user_defined_function('colors_inference', None)

    register_inference_services(aidb_engine, data_dir)
    config = aidb_engine._config
    
    queries={
    # user defined function in SELECT clause
    "test_query_0" : {
      QUERY_STR:
        '''
        SELECT x_min, function1(x_min, y_min), y_max, function2()
        FROM objects00
        WHERE x_min > 600
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, objects00.y_max AS col__2 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'function1',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: [],
           FUNCTION_NAME: 'function2',
           RESULT_COL_NAME: ['function__1']}
        ],
        SELECT_COL: ['col__0', 'function__0', 'col__2', 'function__1'],
        FILTER_PREDICATE: None
        }
      },
    
    # test function with constant parameters
    "test_query_1": {
      QUERY_STR:
        '''
        SELECT x_min, function1(y_min, 2, 3)
        FROM objects00
        WHERE x_min > 600
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, 2 AS col__2, 3 AS col__3 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__1', 'col__2', 'col__3'],
           FUNCTION_NAME: 'function1',
           RESULT_COL_NAME: ['function__0']}
        ],
        SELECT_COL: ['col__0', 'function__0'],
        FILTER_PREDICATE: None
      }
    },

    # user defined function in JOIN clause
    "test_query_2": {
      QUERY_STR:
        '''
        SELECT objects00.frame, x_min, y_max, color
        FROM objects00 JOIN colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE color = 'blue'
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, objects00.x_min AS col__1, objects00.y_max AS col__2, "
        "colors02.color AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (colors02.color = 'blue')",
      DATAFRAME_SQL:{
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__4'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']}
        ],
        SELECT_COL: ['col__0', 'col__1', 'col__2', 'col__3'],
        FILTER_PREDICATE: '(function__0 = TRUE) AND (function__1 = TRUE)'
        }
      },
      
    # user defined function in WHERE clause
    "test_query_3": {
      QUERY_STR:
        '''
        SELECT objects00.frame, x_min, y_max, color
        FROM objects00 JOIN colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE AND sum_function(x_max, y_min) > 1500
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, objects00.x_min AS col__1, objects00.y_max AS col__2, "
        "colors02.color AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.x_max AS col__7, objects00.y_min AS col__8 "
        "FROM objects00 CROSS JOIN colors02",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__4'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']},
          {COL_NAMES: ['col__7', 'col__8'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['function__2']}
        ],
        SELECT_COL: ['col__0', 'col__1', 'col__2', 'col__3'],
        FILTER_PREDICATE: '(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > 1500)'
      }
    },

    # user defined function in SELECT, JOIN, WHERE clause
    "test_query_4": {
      QUERY_STR:
        '''
        SELECT multiply_function(x_min, y_max), color
        FROM objects00 JOIN colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE sum_function(x_min, y_min) > 1500
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__3', 'col__4'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__2']},
          {COL_NAMES: ['col__0', 'col__7'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['function__3']}
        ],
        SELECT_COL: ['function__0', 'col__2'],
        FILTER_PREDICATE: '(function__1 = TRUE) AND (function__2 = TRUE) AND (function__3 > 1500)'
      }
    },

    # OR operator in WHERE clause
    "test_query_5": {
      QUERY_STR:
        '''
        SELECT x_min, y_max, color
        FROM objects00 JOIN colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE sum_function(x_min, y_min) > 1500 OR color = 'blue'
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__3', 'col__4'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']},
          {COL_NAMES: ['col__0', 'col__7'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['function__2']}
        ],
        SELECT_COL: ['col__0', 'col__1', 'col__2'],
        FILTER_PREDICATE: "(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > 1500 OR col__2 = 'blue')"
      }
    },
  
    # comparison between user defined function
    "test_query_6": {
      QUERY_STR:
        '''
        SELECT x_min, y_max, color
        FROM objects00 JOIN colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE sum_function(x_min, y_min) > multiply_function(x_min, y_min)
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__3', 'col__4'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']},
          {COL_NAMES: ['col__0', 'col__7'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['function__2']},
          {COL_NAMES: ['col__0', 'col__7'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__3']}
        ],
        SELECT_COL: ['col__0', 'col__1', 'col__2'],
        FILTER_PREDICATE: "(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > function__3)"
      }
    },
    
    # test user defined function with alias
    "test_query_7": {
      QUERY_STR:
        '''
        SELECT colors_inference(frame, object_id) AS (output1, output2, output3), x_min, y_max
        FROM objects00
        WHERE (x_min > 600 AND output1 LIKE 'blue') OR (y_max < 1000 AND x_max < 1000)
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, objects00.object_id AS col__1, objects00.x_min AS col__2, "
        "objects00.y_max AS col__3, objects00.x_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600 OR objects00.y_max < 1000) AND (objects00.x_min > 600 OR objects00.x_max < 1000)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'colors_inference',
           RESULT_COL_NAME: ['output1', 'output2', 'output3']}
        ],
        SELECT_COL: ['output1', 'output2', 'output3', 'col__2', 'col__3'],
        FILTER_PREDICATE: "(output1 LIKE 'blue' OR col__3 < 1000) AND (output1 LIKE 'blue' OR col__4 < 1000)"
      }
    },
    "test_query_8": {
      QUERY_STR:
        '''
        SELECT colors_inference(frame, object_id) AS (output1, output2, output3), x_min AS col1, y_max AS col2
        FROM objects00
        WHERE (col1 > 600 AND output1 LIKE 'blue') OR (col2 < 1000 AND x_max < 1000)
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, objects00.object_id AS col__1, objects00.x_min AS col__2, "
        "objects00.y_max AS col__3, objects00.x_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600 OR objects00.y_max < 1000) AND (objects00.x_min > 600 OR objects00.x_max < 1000)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'colors_inference',
           RESULT_COL_NAME: ['output1', 'output2', 'output3']}
        ],
        SELECT_COL: ['output1', 'output2', 'output3', 'col__2', 'col__3'],
        FILTER_PREDICATE: "(output1 LIKE 'blue' OR col__3 < 1000) AND (output1 LIKE 'blue' OR col__4 < 1000)"
      }
    },

    # single output user defined function with alias
    "test_query_9": {
      QUERY_STR:
        '''
        SELECT max_function(y_max, y_min) AS output1, power_function(x_min, 2) AS output2, y_min, color
        FROM objects00 join colors02
        WHERE is_equal(objects00.frame, colors02.frame) = TRUE AND is_equal(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800)) AND output1 > 1000 AND output2 > 640000
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.y_max AS col__0, objects00.y_min AS col__1, objects00.x_min AS col__2, 2 AS col__3, "
        "colors02.color AS col__4, objects00.frame AS col__5, colors02.frame AS col__6, objects00.object_id AS col__7, "
        "colors02.object_id AS col__8 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (objects00.x_min > 600 OR objects00.x_max > 600) AND (objects00.x_min > 600 OR objects00.y_min > 800)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'max_function',
           RESULT_COL_NAME: ['output1']},
          {COL_NAMES: ['col__2', 'col__3'],
           FUNCTION_NAME: 'power_function',
           RESULT_COL_NAME: ['output2']},
          {COL_NAMES: ['col__5', 'col__6'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__2']},
          {COL_NAMES: ['col__7', 'col__8'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__3']},
        ],
        SELECT_COL: ['output1', 'output2', 'col__1', 'col__4'],
        FILTER_PREDICATE: "(function__2 = TRUE) AND (function__3 = TRUE) AND (output1 > 1000) AND (output2 > 640000)"
      }
    },
    
    # test user defined functions both within the database and within AIDB
    "test_query_10": {
      QUERY_STR:
        '''
        SELECT multiply_function(x_min, y_min), database_multiply_function(x_min, y_min), x_max, y_max
        FROM objects00
        WHERE x_min > 600 AND y_max < 1000
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, database_multiply_function(objects00.x_min, "
        "objects00.y_min) AS col__2, objects00.x_max AS col__3, objects00.y_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600) AND (objects00.y_max < 1000)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__0', 'col__1'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__0']},
        ],
        SELECT_COL: ['function__0', 'col__2', 'col__3', 'col__4'],
        FILTER_PREDICATE: None
      }
    },
  
    "test_query_11": {
      QUERY_STR:
        '''
        SELECT database_add_function(y_max, x_min), multiply_function(y_min, y_max), color
        FROM objects00 join colors02 ON is_equal(objects00.frame, colors02.frame) = TRUE
            AND is_equal(objects00.object_id, colors02.object_id) = TRUE
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',
      QUERY_EXTRACTED:
        "SELECT database_add_function(objects00.y_max, objects00.x_min) AS col__0, objects00.y_min AS col__1, "
        "objects00.y_max AS col__2, colors02.color AS col__3, objects00.frame AS col__4, colors02.frame AS col__5, "
        "objects00.object_id AS col__6, colors02.object_id AS col__7 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (objects00.x_min > 600 OR objects00.x_max > 600) AND (objects00.x_min > 600 OR objects00.y_min > 800)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__1', 'col__2'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__4', 'col__5'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__1']},
          {COL_NAMES: ['col__6', 'col__7'],
           FUNCTION_NAME: 'is_equal',
           RESULT_COL_NAME: ['function__2']},
        ],
        SELECT_COL: ['col__0', 'function__0', 'col__3'],
        FILTER_PREDICATE: '(function__1 = TRUE) AND (function__2 = TRUE)'
      }
    },    
    
    "test_query_12": {
      QUERY_STR:
        '''
        SELECT frame, database_multiply_function(x_min, y_min), sum_function(x_max, y_max)
        FROM objects00
        WHERE (multiply_function(x_min, y_min) > 400000 AND database_add_function(y_max, x_min) < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, database_multiply_function(objects00.x_min, objects00.y_min) AS col__1, "
        "objects00.x_max AS col__2, objects00.y_max AS col__3, objects00.x_min AS col__4, objects00.y_min AS col__5 "
        "FROM objects00 "
        "WHERE (database_add_function(objects00.y_max, objects00.x_min) < 1600 OR "
        "database_multiply_function(objects00.x_min, objects00.y_min) > 500000)",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__2', 'col__3'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['function__0']},
          {COL_NAMES: ['col__4', 'col__5'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__1']},
        ],
        SELECT_COL: ['col__0', 'col__1', 'function__0'],
        FILTER_PREDICATE: '(function__1 > 400000 OR col__1 > 500000)'
      }
    },
    
    "test_query_13": {
      QUERY_STR:
        '''
        SELECT frame, database_multiply_function(x_min, y_min), sum_function(x_max, y_max) AS output1
        FROM objects00
        WHERE (multiply_function(x_min, y_min) > 400000 AND output1 < 1600)
            OR database_multiply_function(x_min, y_min) > 500000
        ''',
      QUERY_EXTRACTED:
        "SELECT objects00.frame AS col__0, database_multiply_function(objects00.x_min, objects00.y_min) AS col__1, "
        "objects00.x_max AS col__2, objects00.y_max AS col__3, objects00.x_min AS col__4, objects00.y_min AS col__5 "
        "FROM objects00",
      DATAFRAME_SQL: {
        UDF_MAPPING: [
          {COL_NAMES: ['col__2', 'col__3'],
           FUNCTION_NAME: 'sum_function',
           RESULT_COL_NAME: ['output1']},
          {COL_NAMES: ['col__4', 'col__5'],
           FUNCTION_NAME: 'multiply_function',
           RESULT_COL_NAME: ['function__1']},
        ],
        SELECT_COL: ['col__0', 'col__1', 'output1'],
        FILTER_PREDICATE: '(function__1 > 400000 OR col__1 > 500000) AND (output1 < 1600 OR col__1 > 500000)'
      }
    }
  }
    for i in range(len(queries)):
      _test_equality(queries[f'test_query_{i}'], config)


  async def test_invalid_udf_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    aidb_engine.register_user_defined_function('function1', None)
    aidb_engine.register_user_defined_function('function2', None)

    register_inference_services(aidb_engine, data_dir)
    config = aidb_engine._config

    invalid_query_str = [
      '''SELECT function1(AVG(x_min), AVG(y_min)) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%;''',
      '''SELECT function1(x_min) FROM objects00 RECALL_TARGET 90% CONFIDENCE 95%;''',
      '''SELECT function1(x_min) FROM objects00 LIMIT 100;''',
      '''SELECT function1(x_min) FROM objects00 WHERE y_min > (SELECT AVG(y_min) FROM objects00);''',
      '''SELECT function1(function2(x_min)) FROM objects00 WHERE y_min > (SELECT AVG(y_min) FROM objects00);''',
      '''SELECT function1(SUM(x_min), SUM(y_max)) FROM objects00;'''
    ]
    for query_str in invalid_query_str:
      query = Query(query_str, config)
      assert query.is_udf_query == True
      with self.assertRaises(Exception):
       query.check_udf_query_validity()


if __name__ == '__main__':
  unittest.main()
