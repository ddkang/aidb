import os
import unittest
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from unittest import IsolatedAsyncioTestCase

from aidb.query.query import Query
from tests.inference_service_utils.inference_service_setup import \
    register_inference_services
from tests.utils import setup_gt_and_aidb_engine

DB_URL = "sqlite+aiosqlite://"

# normal query dataclass
@dataclass
class TestQuery:
  query_str: str
  normalized_query_str: str
  correct_fp: list
  correct_service: list
  correct_tables: list
  num_of_select_clauses: int
    
    
  def are_lists_equal(self, list1, list2):
    for sub_list1, sub_list2 in zip(list1, list2):
      assert Counter(sub_list1) == Counter(sub_list2)
      
      
  def _test_query(self, config):
    query = Query(self.query_str, config)
    # test the number of queries
    assert len(query.all_queries_in_expressions) == self.num_of_select_clauses
    assert query.query_after_normalizing.sql_str == self.normalized_query_str
    and_fp = []
    for and_connected in query.filtering_predicates:
      or_fp = []
      for or_connected in and_connected:
        or_fp.append(or_connected.sql())
      and_fp.append(or_fp)
    self.are_lists_equal(and_fp, self.correct_fp)
    self.are_lists_equal(query.inference_engines_required_for_filtering_predicates, self.correct_service)
    self.are_lists_equal(query.tables_in_filtering_predicates, self.correct_tables)  


# udf query dataclasses       
@dataclass
class UdfMapping:
  col_names: list
  function_name: str
  result_col_name: list


@dataclass
class DataframeSql:
  udf_mapping: list
  select_col: list
  filter_predicate: str


@dataclass
class UdfTestQuery:
  query_str: str
  query_after_extraction: str
  dataframe_sql: DataframeSql


  def _test_equality(self, config):
    query = Query(self.query_str, config)
    dataframe_sql, query_after_extraction = query.udf_query
    assert query_after_extraction.sql_str==self.query_after_extraction
    assert len(dataframe_sql['udf_mapping']) == len(self.dataframe_sql.udf_mapping)
    # unpack dict values into dataclass and verify that instance values are equal
    assert all (any((e1==UdfMapping(**e2)) for e2 in dataframe_sql['udf_mapping']) 
                for e1 in self.dataframe_sql.udf_mapping)
    assert dataframe_sql['select_col'] == self.dataframe_sql.select_col
    filter_predicate = query.convert_and_connected_fp_to_exp(dataframe_sql['filter_predicate'])
    if filter_predicate:
      filter_predicate = filter_predicate.sql()  
    assert filter_predicate == self.dataframe_sql.filter_predicate


@dataclass
class QueryFilteringPredicatesParsing:
  query_str: str
  correct_fp: str

  def _test_filtering_predicates(self, config):
    query = Query(self.query_str, config)
    and_connected = []
    filtering_predicates = query.filtering_predicates
    for fp in filtering_predicates:
      and_connected.append(' OR '.join([p.sql() for p in fp]))
    and_connected = [f'({fp})' for fp in and_connected]
    parsed_fp = ' AND '.join(and_connected)
    if parsed_fp != self.correct_fp:
      raise AssertionError(f"Failed to parse query: '{self.query_str}'. \n"
                           f"Parsed fp: '{parsed_fp}', but the expected type is '{self.correct_fp}'.")

          
class QueryParsingTests(IsolatedAsyncioTestCase):

  async def test_nested_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config
    
    queries = {
    "test_query_0": TestQuery(
        '''
        SELECT color as alias_color
        FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame
        WHERE alias_color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2)
        OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100);
        ''',
        ("SELECT colors02.color "
         "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
         "WHERE colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2) "
         "OR colors02.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)"),
        # test replacing column with root column in filtering predicate,
        # the root column of 'colors02.object_id' is 'objects00'
        [['colors02.color IN (SELECT table2.color AS alias_color2 FROM colors02 AS table2)',
          'objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 100)']],    
        # filter predicates connected by OR are in same set
        [{'colors02', 'objects00'}],
        [{'colors02', 'objects00'}],            
        3
    ),
    # test table alias
    "test_query_1": TestQuery(
        '''
        SELECT table1.color 
        FROM colors02 AS table1 WHERE frame IN (SELECT * FROM blobs_00)
        AND object_id > 0
        ''',
        # test table alias
        ("SELECT colors02.color FROM colors02 "
         "WHERE colors02.frame IN (SELECT * FROM blobs_00) "
         "AND colors02.object_id > 0"),
        # test replacing column with root column in filtering predicate,
        # the root column of 'colors02.object_id' is 'objects00'
        [['blobs_00.frame IN (SELECT * FROM blobs_00)'],
         ['objects00.object_id > 0']],
        # filter predicates connected by AND are in different set 
        [set(), {'objects00'}],
        [set(), {'objects00'}],
        2
    ),
    # test sub-subquery
    "test_query_2": TestQuery(
        '''
        SELECT frame, object_id 
        FROM colors02 AS cl
        WHERE cl.object_id > (SELECT AVG(object_id) FROM objects00
        WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))
        ''',
        ("SELECT colors02.frame, colors02.object_id FROM colors02 "
         "WHERE colors02.object_id > (SELECT AVG(object_id) FROM objects00 WHERE frame > "
         "(SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))"),
        [["objects00.object_id > (SELECT AVG(object_id) FROM objects00 " 
          "WHERE frame > (SELECT AVG(frame) FROM blobs_00 WHERE frame > 500))"]],
        [{'objects00'}], 
        [{'objects00'}], 
        3
    ),
    # test multiple aliases
    "test_query_3": TestQuery(
        '''
        SELECT color, table2.x_min
        FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
        WHERE color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3)
        AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
        ''', 
        ("SELECT colors02.color, objects00.x_min "
         "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
         "WHERE colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
         "AND objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"),
        [["colors02.color IN (SELECT table3.color AS alias_color2 FROM colors02 AS table3)"],
         ["objects00.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]],
        [{'colors02'}, {'objects00'}],
        [{'colors02'}, {'objects00'}],
        3
    ),
    # comparison between subquery
    "test_query_4": TestQuery(
        '''
        SELECT color, table2.x_min
        FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
        WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3)
        > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
        ''',
        ("SELECT colors02.color, objects00.x_min "
         "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
         "WHERE (SELECT table3.color AS alias_color2 FROM colors02 AS table3) "
         "> (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"),
        [["(SELECT table3.color AS alias_color2 FROM colors02 AS table3) > "
        "(SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500)"]],
        [{}],
        [{}],
        3
    ),
    "test_query_5": TestQuery(
        '''
        SELECT color AS col1, table2.x_min AS col2, table2.y_min AS col3
        FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame;
        ''',
        ("SELECT colors02.color, objects00.x_min, objects00.y_min "
         "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame"),
        [[]],
        [{}], 
        [{}],
        1
    ),
    "test_query_6": TestQuery( 
        '''
        SELECT color AS col1, table2.x_min AS col2, table3.frame AS col3
        FROM colors02 table1 LEFT JOIN objects00 table2 ON table1.frame = table2.frame
        JOIN blobs_00 table3 ON table2.frame = table3.frame;
        ''',
        ("SELECT colors02.color, objects00.x_min, blobs_00.frame "
         "FROM colors02 LEFT JOIN objects00 ON colors02.frame = objects00.frame "
         "JOIN blobs_00 ON objects00.frame = blobs_00.frame"),
        [[]],
        [{}],
        [{}],
        1
    ),
    "test_query_7": TestQuery(
        '''
        SELECT color, x_min AS col2, colors02.frame AS col3
        FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
        WHERE color = 'blue' AND x_min > 600;
        ''',
        ("SELECT colors02.color, objects00.x_min, colors02.frame "
         "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
         "WHERE colors02.color = 'blue' AND objects00.x_min > 600"),
        [["colors02.color = 'blue'"], 
         ['objects00.x_min > 600']],
        [{'colors02'}, {'objects00'}],
        [{'colors02'}, {'objects00'}],
        1
    ),
    "test_query_8": TestQuery(
        '''
        SELECT color, USERFUNCTION(x_min, y_min, x_max, y_max)
        FROM colors02 JOIN objects00 table2 ON colors02.frame = table2.frame
        WHERE table2.frame > 10000 OR y_max < 800;
        ''',
        ("SELECT colors02.color, USERFUNCTION(objects00.x_min, objects00.y_min, "
         "objects00.x_max, objects00.y_max) "
         "FROM colors02 JOIN objects00 ON colors02.frame = objects00.frame "
         "WHERE objects00.frame > 10000 OR objects00.y_max < 800"),
        [['blobs_00.frame > 10000', 'objects00.y_max < 800']], 
        [{'objects00'}],
        [{'objects00'}],
        1
    )
}
    for i in range(len(queries)):
      queries[f'test_query_{i}']._test_query(config)


  # We don't support using approximate query as a subquery
  async def test_approximate_query_as_subquery(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    register_inference_services(aidb_engine, data_dir)

    config = aidb_engine._config
    queries = {
      # approx aggregation as a subquery
      "query_str0": 
          '''
          SELECT x_min 
          FROM  objects00
          WHERE x_min > (SELECT AVG(x_min) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%)
          AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
          ''',
      # approx select as a subquery
      "query_str1": 
          '''
          SELECT x_min 
          FROM  objects00
          WHERE frame IN (SELECT frame FROM colors02 where color LIKE 'blue'
          RECALL_TARGET {RECALL_TARGET}% CONFIDENCE 95%;)
          AND table2.object_id > (SELECT AVG(ob.object_id) FROM objects00 AS ob WHERE frame > 500);
          '''
    }
    for i in range(len(queries)):
      parsed_query = Query(queries[f'query_str{i}'], config)
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
    "test_query_0": TestQuery(
        '''
        SELECT AVG(x_min) 
        FROM  objects00
        WHERE frame > (SELECT AVG(frame) FROM blobs_00)
        ERROR_TARGET 10% CONFIDENCE 95%;
        ''',
        ("SELECT AVG(objects00.x_min) "
         "FROM objects00 "
         "WHERE objects00.frame > (SELECT AVG(frame) FROM blobs_00) ERROR_TARGET 10% CONFIDENCE 95%"),
        [["blobs_00.frame > (SELECT AVG(frame) FROM blobs_00)"]],
        # filter predicates connected by OR are in same set
        [{}],
        [{}], 
        2
    ),
    "test_query_1": TestQuery(
        '''
        SELECT frame 
        FROM colors02
        WHERE color IN (SELECT color FROM colors02 WHERE frame > 10000)
        RECALL_TARGET 80%
        CONFIDENCE 95%;
        ''',
        ("SELECT colors02.frame FROM colors02 "
         "WHERE colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000) "
         "RECALL_TARGET 80% "
         "CONFIDENCE 95%"),
        [["colors02.color IN (SELECT color FROM colors02 WHERE frame > 10000)"]],
        [{'colors02'}],
        [{'colors02'}],
        2
    )
}
    for i in range(len(queries)):
      queries[f'test_query_{i}']._test_query(config)

  async def test_udf_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    aidb_engine.register_user_defined_function('SUM_FUNCTION', None)
    aidb_engine.register_user_defined_function('IS_EQUAL', None)
    aidb_engine.register_user_defined_function('POWER_FUNCTION', None)
    aidb_engine.register_user_defined_function('MAX_FUNCTION', None)
    aidb_engine.register_user_defined_function('MULTIPLY_FUNCTION', None)
    aidb_engine.register_user_defined_function('FUNCTION1', None)
    aidb_engine.register_user_defined_function('FUNCTION2', None)
    aidb_engine.register_user_defined_function('COLORS_INFERENCE', None)

    register_inference_services(aidb_engine, data_dir)
    config = aidb_engine._config

    queries={
    # user defined function in SELECT clause
    "test_query_0" : UdfTestQuery( 
        '''
        SELECT x_min, FUNCTION1(x_min, y_min), y_max, FUNCTION2()
        FROM objects00
        WHERE x_min > 600
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, objects00.y_max AS col__2 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600)",
        DataframeSql(
            [UdfMapping(
                ['col__0', 'col__1'],
                'FUNCTION1',
                ['function__0']),
             UdfMapping( 
                [],
                'FUNCTION2',
                ['function__1'])
            ],
            ['col__0', 'function__0', 'col__2', 'function__1'],
            None
        )
    ),
    
    # test function with constant parameters
    "test_query_1": UdfTestQuery(
        '''
        SELECT x_min, FUNCTION1(y_min, 2, 3)
        FROM objects00
        WHERE x_min > 600
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, 2 AS col__2, 3 AS col__3 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600)",
        DataframeSql(
            [UdfMapping( 
                ['col__1', 'col__2', 'col__3'],
                'FUNCTION1',
                ['function__0'])
            ],
            ['col__0', 'function__0'],
            None
        )
    ),

    # user defined function in JOIN clause
    "test_query_2": UdfTestQuery(
        '''
        SELECT objects00.frame, x_min, y_max, color
        FROM objects00 JOIN colors02 ON IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE
        WHERE color = 'blue'
        ''',
        "SELECT objects00.frame AS col__0, objects00.x_min AS col__1, objects00.y_max AS col__2, "
        "colors02.color AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (colors02.color = 'blue')",
        DataframeSql(
            [UdfMapping(
                ['col__0', 'col__4'],
                'IS_EQUAL',
                ['function__0']),
            UdfMapping(
                ['col__5', 'col__6'],
                'IS_EQUAL', 
                ['function__1'])
            ],
            ['col__0', 'col__1', 'col__2', 'col__3'],
            '(function__0 = TRUE) AND (function__1 = TRUE)'
        )
    ),

    # user defined function in WHERE clause
    "test_query_3": UdfTestQuery(
        '''
        SELECT objects00.frame, x_min, y_max, color
        FROM objects00 JOIN colors02
        WHERE IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE AND SUM_FUNCTION(x_max, y_min) > 1500
        ''',
        "SELECT objects00.frame AS col__0, objects00.x_min AS col__1, objects00.y_max AS col__2, "
        "colors02.color AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.x_max AS col__7, objects00.y_min AS col__8 "
        "FROM objects00 CROSS JOIN colors02",
        DataframeSql(
            [UdfMapping(
                ['col__0', 'col__4'],
                'IS_EQUAL',
                ['function__0']),
            UdfMapping(
                ['col__5', 'col__6'],
                'IS_EQUAL',
                ['function__1']),
            UdfMapping(
                ['col__7', 'col__8'],
                'SUM_FUNCTION', 
                ['function__2'])
            ],
            ['col__0', 'col__1', 'col__2', 'col__3'],
            '(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > 1500)'
        )
    ),

    # user defined function in SELECT, JOIN, WHERE clause
    "test_query_4": UdfTestQuery(
        '''
        SELECT MULTIPLY_FUNCTION(x_min, y_max), color
        FROM objects00 JOIN colors02 ON IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE
        WHERE SUM_FUNCTION(x_min, y_min) > 1500
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
        DataframeSql(
            [UdfMapping(
                  ['col__0', 'col__1'],
                  'MULTIPLY_FUNCTION',
                  ['function__0']),
             UdfMapping(
                  ['col__3', 'col__4'],
                  'IS_EQUAL',
                  ['function__1']),
             UdfMapping(
                  ['col__5', 'col__6'],
                  'IS_EQUAL',
                  ['function__2']),
             UdfMapping(
                  ['col__0', 'col__7'],
                  'SUM_FUNCTION',
                  ['function__3'])
            ],
            ['function__0', 'col__2'],
            '(function__1 = TRUE) AND (function__2 = TRUE) AND (function__3 > 1500)'
        ) 
    ),

    # OR operator in WHERE clause
    "test_query_5": UdfTestQuery(
        '''
        SELECT x_min, y_max, color
        FROM objects00 JOIN colors02 ON IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE
        WHERE SUM_FUNCTION(x_min, y_min) > 1500 OR color = 'blue'
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
        DataframeSql(
            [UdfMapping(
                ['col__3', 'col__4'],
                'IS_EQUAL',
                ['function__0']),
            UdfMapping(        
                ['col__5', 'col__6'],
                'IS_EQUAL',
                ['function__1']),
            UdfMapping(
                ['col__0', 'col__7'],
                'SUM_FUNCTION',
                ['function__2'])
            ], 
            ['col__0', 'col__1', 'col__2'],
            "(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > 1500 OR col__2 = 'blue')"
        )
    ),
  
    # comparison between user defined function
    "test_query_6": UdfTestQuery(
        '''
        SELECT x_min, y_max, color
        FROM objects00 JOIN colors02 ON IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE
        WHERE SUM_FUNCTION(x_min, y_min) > MULTIPLY_FUNCTION(x_min, y_min)
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_max AS col__1, colors02.color AS col__2, "
        "objects00.frame AS col__3, colors02.frame AS col__4, objects00.object_id AS col__5, colors02.object_id AS col__6, "
        "objects00.y_min AS col__7 "
        "FROM objects00 CROSS JOIN colors02",
        DataframeSql(
            [UdfMapping( 
                ['col__3', 'col__4'],
                'IS_EQUAL',
                ['function__0']),
            UdfMapping( 
                ['col__5', 'col__6'],
                'IS_EQUAL',
                ['function__1']),
            UdfMapping( 
                ['col__0', 'col__7'],
                'SUM_FUNCTION',
                ['function__2']),
            UdfMapping( 
                ['col__0', 'col__7'],
                'MULTIPLY_FUNCTION',
                ['function__3'])
            ],
            ['col__0', 'col__1', 'col__2'], 
            "(function__0 = TRUE) AND (function__1 = TRUE) AND (function__2 > function__3)"
        )
    ),

    # test user defined function with alias
    "test_query_7": UdfTestQuery(
        '''
        SELECT COLORS_INFERENCE(frame, object_id) AS (output1, output2, output3), x_min, y_max
        FROM objects00
        WHERE (x_min > 600 AND output1 LIKE 'blue') OR (y_max < 1000 AND x_max < 1000)
        ''',
        "SELECT objects00.frame AS col__0, objects00.object_id AS col__1, objects00.x_min AS col__2, "
        "objects00.y_max AS col__3, objects00.x_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600 OR objects00.y_max < 1000) AND (objects00.x_min > 600 OR objects00.x_max < 1000)",
        DataframeSql(
            [UdfMapping( 
                ['col__0', 'col__1'],
                'COLORS_INFERENCE',
                ['output1', 'output2', 'output3'])
            ],
            ['output1', 'output2', 'output3', 'col__2', 'col__3'],
            "(output1 LIKE 'blue' OR col__3 < 1000) AND (output1 LIKE 'blue' OR col__4 < 1000)"
        )
    ),
    "test_query_8": UdfTestQuery(
        '''
        SELECT COLORS_INFERENCE(frame, object_id) AS (output1, output2, output3), x_min AS col1, y_max AS col2
        FROM objects00
        WHERE (col1 > 600 AND output1 LIKE 'blue') OR (col2 < 1000 AND x_max < 1000)
        ''',
        "SELECT objects00.frame AS col__0, objects00.object_id AS col__1, objects00.x_min AS col__2, "
        "objects00.y_max AS col__3, objects00.x_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600 OR objects00.y_max < 1000) AND (objects00.x_min > 600 OR objects00.x_max < 1000)",
        DataframeSql(
            [UdfMapping( 
                ['col__0', 'col__1'], 
                'COLORS_INFERENCE',
                ['output1', 'output2', 'output3'])
            ], 
            ['output1', 'output2', 'output3', 'col__2', 'col__3'],
            "(output1 LIKE 'blue' OR col__3 < 1000) AND (output1 LIKE 'blue' OR col__4 < 1000)"
        )
    ),

    # single output user defined function with alias
    "test_query_9": UdfTestQuery(
        '''
        SELECT MAX_FUNCTION(y_max, y_min) AS output1, POWER_FUNCTION(x_min, 2) AS output2, y_min, color
        FROM objects00 join colors02
        WHERE IS_EQUAL(objects00.frame, colors02.frame) = TRUE AND IS_EQUAL(objects00.object_id, colors02.object_id)
            = TRUE AND (x_min > 600 OR (x_max >600 AND y_min > 800)) AND output1 > 1000 AND output2 > 640000
        ''',
        "SELECT objects00.y_max AS col__0, objects00.y_min AS col__1, objects00.x_min AS col__2, 2 AS col__3, "
        "colors02.color AS col__4, objects00.frame AS col__5, colors02.frame AS col__6, objects00.object_id AS col__7, "
        "colors02.object_id AS col__8 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (objects00.x_min > 600 OR objects00.x_max > 600) AND (objects00.x_min > 600 OR objects00.y_min > 800)",
        DataframeSql(
           [UdfMapping(
               ['col__0', 'col__1'],
               'MAX_FUNCTION',
               ['output1']),
            UdfMapping(        
               ['col__2', 'col__3'],
               'POWER_FUNCTION',
               ['output2']),
            UdfMapping(
               ['col__5', 'col__6'],
               'IS_EQUAL',
               ['function__2']),
            UdfMapping(
               ['col__7', 'col__8'],
               'IS_EQUAL',
               ['function__3']),
           ], 
           ['output1', 'output2', 'col__1', 'col__4'],
           "(function__2 = TRUE) AND (function__3 = TRUE) AND (output1 > 1000) AND (output2 > 640000)"
         )
    ),
    
    # test user defined functions both within the database and within AIDB
    "test_query_10": UdfTestQuery(
        '''
        SELECT MULTIPLY_FUNCTION(x_min, y_min), DATABASE_MULTIPLY_FUNCTION(x_min, y_min), x_max, y_max
        FROM objects00
        WHERE x_min > 600 AND y_max < 1000
        ''',
        "SELECT objects00.x_min AS col__0, objects00.y_min AS col__1, DATABASE_MULTIPLY_FUNCTION(objects00.x_min, "
        "objects00.y_min) AS col__2, objects00.x_max AS col__3, objects00.y_max AS col__4 "
        "FROM objects00 "
        "WHERE (objects00.x_min > 600) AND (objects00.y_max < 1000)",
        DataframeSql(
            [UdfMapping(
                ['col__0', 'col__1'], 
                'MULTIPLY_FUNCTION',
                ['function__0'])
            ],
            ['function__0', 'col__2', 'col__3', 'col__4'],
            None
        )
    ),
    
  
    "test_query_11": UdfTestQuery(
        '''
        SELECT DATABASE_ADD_FUNCTION(y_max, x_min), MULTIPLY_FUNCTION(y_min, y_max), color
        FROM objects00 join colors02 ON IS_EQUAL(objects00.frame, colors02.frame) = TRUE
            AND IS_EQUAL(objects00.object_id, colors02.object_id) = TRUE
        WHERE x_min > 600 OR (x_max >600 AND y_min > 800)
        ''',
        "SELECT DATABASE_ADD_FUNCTION(objects00.y_max, objects00.x_min) AS col__0, objects00.y_min AS col__1, "
        "objects00.y_max AS col__2, colors02.color AS col__3, objects00.frame AS col__4, colors02.frame AS col__5, "
        "objects00.object_id AS col__6, colors02.object_id AS col__7 "
        "FROM objects00 CROSS JOIN colors02 "
        "WHERE (objects00.x_min > 600 OR objects00.x_max > 600) AND (objects00.x_min > 600 OR objects00.y_min > 800)",
        DataframeSql(
            [UdfMapping(
                ['col__1', 'col__2'],
                'MULTIPLY_FUNCTION',
                ['function__0']),
             UdfMapping(
                ['col__4', 'col__5'],
                'IS_EQUAL',
                ['function__1']),
             UdfMapping(
                ['col__6', 'col__7'],
                'IS_EQUAL',
                ['function__2']),
            ],
                ['col__0', 'function__0', 'col__3'],
                '(function__1 = TRUE) AND (function__2 = TRUE)'
        )
    ),    
    
    "test_query_12": UdfTestQuery(
        '''
        SELECT frame, DATABASE_MULTIPLY_FUNCTION(x_min, y_min), SUM_FUNCTION(x_max, y_max)
        FROM objects00
        WHERE (MULTIPLY_FUNCTION(x_min, y_min) > 400000 AND DATABASE_ADD_FUNCTION(y_max, x_min) < 1600)
            OR DATABASE_MULTIPLY_FUNCTION(x_min, y_min) > 500000
        ''',
        "SELECT objects00.frame AS col__0, DATABASE_MULTIPLY_FUNCTION(objects00.x_min, objects00.y_min) AS col__1, "
        "objects00.x_max AS col__2, objects00.y_max AS col__3, objects00.x_min AS col__4, objects00.y_min AS col__5 "
        "FROM objects00 "
        "WHERE (DATABASE_ADD_FUNCTION(objects00.y_max, objects00.x_min) < 1600 OR "
        "DATABASE_MULTIPLY_FUNCTION(objects00.x_min, objects00.y_min) > 500000)",
        DataframeSql(
            [UdfMapping(
                ['col__2', 'col__3'],
                'SUM_FUNCTION',
                ['function__0']),
             UdfMapping(
                ['col__4', 'col__5'],
                'MULTIPLY_FUNCTION',
                ['function__1']),
            ],
              ['col__0', 'col__1', 'function__0'],
              '(function__1 > 400000 OR col__1 > 500000)'
        )
    ),
    
    "test_query_13": UdfTestQuery(
        '''
        SELECT frame, DATABASE_MULTIPLY_FUNCTION(x_min, y_min), SUM_FUNCTION(x_max, y_max) AS output1
        FROM objects00
        WHERE (MULTIPLY_FUNCTION(x_min, y_min) > 400000 AND output1 < 1600)
            OR DATABASE_MULTIPLY_FUNCTION(x_min, y_min) > 500000
        ''',
        "SELECT objects00.frame AS col__0, DATABASE_MULTIPLY_FUNCTION(objects00.x_min, objects00.y_min) AS col__1, "
        "objects00.x_max AS col__2, objects00.y_max AS col__3, objects00.x_min AS col__4, objects00.y_min AS col__5 "
        "FROM objects00",
        DataframeSql(
            [UdfMapping(
                ['col__2', 'col__3'],
                'SUM_FUNCTION',
                ['output1']),
              UdfMapping(
                ['col__4', 'col__5'],
                'MULTIPLY_FUNCTION',
                ['function__1']),
              ],
                ['col__0', 'col__1', 'output1'], 
                '(function__1 > 400000 OR col__1 > 500000) AND (output1 < 1600 OR col__1 > 500000)'
        )
    )
  }
    for i in range(len(queries)):
      queries[f'test_query_{i}']._test_equality(config)


  async def test_invalid_udf_query(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)

    aidb_engine.register_user_defined_function('FUNCTION1', None)
    aidb_engine.register_user_defined_function('FUNCTION2', None)

    register_inference_services(aidb_engine, data_dir)
    config = aidb_engine._config

    invalid_query_str = [
      '''SELECT FUNCTION1(AVG(x_min), AVG(y_min)) FROM objects00 ERROR_TARGET 10% CONFIDENCE 95%;''',
      '''SELECT FUNCTION1(x_min) FROM objects00 RECALL_TARGET 90% CONFIDENCE 95%;''',
      '''SELECT FUNCTION1(x_min) FROM objects00 LIMIT 100;''',
      '''SELECT FUNCTION1(x_min) FROM objects00 WHERE y_min > (SELECT AVG(y_min) FROM objects00);''',
      '''SELECT FUNCTION1(FUNCTION2(x_min)) FROM objects00 WHERE y_min > (SELECT AVG(y_min) FROM objects00);''',
      '''SELECT FUNCTION1(SUM(x_min), SUM(y_max)) FROM objects00;'''
    ]
    for query_str in invalid_query_str:
      query = Query(query_str, config)
      assert query.is_udf_query == True
      with self.assertRaises(Exception):
       query.check_udf_query_validity()


  async def test_filtering_predicates(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/jackson')
    gt_engine, aidb_engine = await setup_gt_and_aidb_engine(DB_URL, data_dir)
    config = aidb_engine._config

    queries = {
      'test_query_0': QueryFilteringPredicatesParsing(
        '''
        SELECT *
        FROM objects00
        WHERE x_min > 600 OR objects00.frame > 700 AND objects00.frame < 1000 OR x_max = 1
        ''',
        "(objects00.x_min > 600 OR blobs_00.frame > 700 OR objects00.x_max = 1) "
        "AND (objects00.x_min > 600 OR blobs_00.frame < 1000 OR objects00.x_max = 1)"
      ),
      'test_query_1': QueryFilteringPredicatesParsing(
        '''
        SELECT *
        FROM objects00
        WHERE (y_min < 300 OR y_max > 700) AND (frame >= 500 AND frame <= 600) OR x_min = 100
        ''',
        "(blobs_00.frame >= 500 OR objects00.x_min = 100) "
        "AND (blobs_00.frame <= 600 OR objects00.x_min = 100) "
        "AND (objects00.y_min < 300 OR objects00.y_max > 700 OR objects00.x_min = 100)"
      ),
      'test_query_2': QueryFilteringPredicatesParsing(
        '''
        SELECT *
        FROM objects00
        WHERE (x_min < 200 OR y_max > 800) AND frame < 400 OR NOT (x_max = 900)
        ''',
        "(blobs_00.frame < 400 OR NOT objects00.x_max = 900) "
        "AND (objects00.x_min < 200 OR objects00.y_max > 800 OR NOT objects00.x_max = 900)"
      ),
      'test_query_3': QueryFilteringPredicatesParsing(
        '''
        SELECT *
        FROM objects00
        WHERE (x_max > 500 OR (y_min < 250 AND y_max > 750)) AND NOT (frame >= 300 OR frame <= 800)   
        ''',
        "(NOT blobs_00.frame >= 300) "
        "AND (NOT blobs_00.frame <= 800) "
        "AND (objects00.x_max > 500 OR objects00.y_min < 250) "
        "AND (objects00.x_max > 500 OR objects00.y_max > 750)"
      ),
      'test_query_4': QueryFilteringPredicatesParsing(
        '''
        SELECT *
        FROM objects00
        WHERE (frame >= 100 AND frame <= 300) OR (x_min < 100 AND y_min > 600) AND NOT (x_max = y_max)
        ''',
        "(blobs_00.frame >= 100 OR objects00.x_min < 100) "
        "AND (blobs_00.frame >= 100 OR objects00.y_min > 600) "
        "AND (blobs_00.frame <= 300 OR objects00.x_min < 100) "
        "AND (blobs_00.frame <= 300 OR objects00.y_min > 600) "
        "AND (blobs_00.frame >= 100 OR NOT objects00.x_max = objects00.y_max) "
        "AND (blobs_00.frame <= 300 OR NOT objects00.x_max = objects00.y_max)"
      )
    }
    for i in range(len(queries)):
      queries[f'test_query_{i}']._test_filtering_predicates(config)


if __name__ == '__main__':
  unittest.main()
