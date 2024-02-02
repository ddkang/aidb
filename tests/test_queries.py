import unittest
from enum import Enum

from aidb.query.query import Query

# Valid
valid_avg_sql = '''
SELECT AVG(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95%;
'''

valid_count_sql = '''
SELECT COUNT(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95%;
'''

valid_sum_sql = '''
SELECT SUM(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95%;
'''

# Invalid
unsupported_agg_aqp_sql = '''
SELECT MAX(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95%;
'''

agg_no_et_sql = '''
SELECT AVG(bar)
FROM foo
CONFIDENCE 95%;
'''

agg_no_conf_sql = '''
SELECT AVG(bar)
FROM foo
ERROR_TARGET 1%;
'''

not_select_sql = '''
INSERT INTO employees (id, name)
VALUES (101, 'John');
'''

approximate_agg_group_by_sql = '''
SELECT AVG(bar)
FROM foo
GROUP BY (frame)
ERROR_TARGET 1%
CONFIDENCE 95%;
'''

approximate_agg_join_sql = '''
SELECT AVG(bar)
FROM foo join joo
on foo.x = joo.x
ERROR_TARGET 1%
CONFIDENCE 95%;
'''


class AQPKeywordTests(unittest.TestCase):
  def test_confidence(self):
    query = Query(valid_avg_sql, None)
    confidence = query.confidence
    self.assertEqual(confidence, 95)

  def test_error_target(self):
    query = Query(valid_avg_sql, None)
    error_target = query.error_target
    self.assertEqual(error_target, 0.01)


class AQPValidationTests(unittest.TestCase):
  # Valid queries
  def test_valid_aqp(self):
    query = Query(valid_avg_sql, None)
    self.assertEqual(query.is_valid_aqp_query(), True)


  def test_valid_aqp_join(self):
    query = Query(approximate_agg_join_sql, None)
    self.assertEqual(query.is_valid_aqp_query(), True)


  # Below is a list of tests designed to assess invalid queries
  def test_invalid_count_sql(self):
    query = Query(valid_count_sql, None)
    self.assertEqual(query.is_valid_aqp_query(), True)


  def test_invalid_sum_sql(self):
    query = Query(valid_sum_sql, None)
    self.assertEqual(query.is_valid_aqp_query(), True)


  def test_invalid_unsupported_agg_aqp(self):
    query = Query(unsupported_agg_aqp_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()

  def test_invalid_agg_no_et(self):
    query = Query(agg_no_et_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()

  def test_invalid_agg_no_conf(self):
    query = Query(agg_no_conf_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()

  def test_invalid_not_select_sql(self):
    query = Query(not_select_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()

  def test_invalid_approximate_agg_group_by_sql(self):
    query = Query(approximate_agg_group_by_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()


# This class tests the correctness of the parsed type for a query.
class QueryTypeTests(unittest.TestCase):
  class QueryType(Enum):
    APPROX_AGG = 'approx_agg'
    APPROX_AGG_JOIN = 'approx_agg_join'
    APPROX_SELECT = 'approx_select'
    LIMIT_QUERY = 'limit_query'
    SELECT_QUERY = 'select_query'
    NON_SELECT = 'none_select'


  def _test_query_type_equality(self, query_str, query_type):
    query = Query(query_str, None)
    if query.is_approx_agg_query:
      if query.is_aqp_join_query:
        parsed_type = self.QueryType.APPROX_AGG_JOIN
      else:
        parsed_type = self.QueryType.APPROX_AGG
    elif query.is_approx_select_query:
      parsed_type = self.QueryType.APPROX_SELECT
    elif query.is_limit_query():
      parsed_type = self.QueryType.LIMIT_QUERY
    elif query.is_select_query():
      parsed_type = self.QueryType.SELECT_QUERY
    else:
      parsed_type = self.QueryType.NON_SELECT

    if parsed_type != query_type:
      raise AssertionError(f"Failed to parse query: '{query_str}'. \n"
                           f"Parsed type: '{parsed_type}', but the expected type is '{query_type}'.")


  def test_query_type(self):
    approx_agg_sql = [
      (
        self.QueryType.APPROX_AGG,
        '''
        SELECT AVG(col1)
        FROM table1  
        ERROR_TARGET 5%
        CONFIDENCE 95%;
        ''',
      ),
      (
        self.QueryType.APPROX_AGG_JOIN,
        '''
        SELECT COUNT(*)
        FROM table1 CROSS JOIN table2
        WHERE function1(table1.col1, table2.col1) = TRUE
        ERROR_TARGET 5%
        CONFIDENCE 95%;
        ''',
      ),
      (
        self.QueryType.APPROX_SELECT,
        '''
        SELECT col1
        FROM table1
        RECALL_TARGET 10%
        CONFIDENCE 95%;
        ''',
      ),
      (
        self.QueryType.LIMIT_QUERY,
        '''
        SELECT col1
        FROM table1
        LIMIT 1000;
        ''',
      ),
      (
        self.QueryType.SELECT_QUERY,
        '''
        SELECT col1
        FROM table1
        WHERE col2 > 1000;
        ''',
      ),
      (
        self.QueryType.SELECT_QUERY,
        '''
        SELECT AVG(table1.t1_col1), COUNT(table2.t2_col2)
        FROM table1 JOIN table2
        ON table1.t1_col1 = table2.t2_col1
        ''',
      ),
      (
        self.QueryType.NON_SELECT,
        '''
        CREATE TABLE Employees (
        EmployeeID INT PRIMARY KEY,
        FirstName VARCHAR(50),
        LastName VARCHAR(50),
        );
        ''',
      ),
    ]

    for query_type, query_str in approx_agg_sql:
      self._test_query_type_equality(query_str, query_type)


if __name__ == '__main__':
  unittest.main()
