import unittest
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
CONFIDENCE 95;
'''

approximate_agg_join_sql = '''
SELECT AVG(bar)
FROM foo join joo
on foo.x = joo.x
ERROR_TARGET 1%
CONFIDENCE 95;
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
  def test_valid_aqp(self):
    query = Query(valid_avg_sql, None)
    self.assertEqual(query.is_valid_aqp_query(), True)


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

  def test_invalid_approximate_agg_join_sql(self):
    query = Query(approximate_agg_join_sql, None)
    with self.assertRaises(Exception):
      query.is_valid_aqp_query()


if __name__ == '__main__':
  unittest.main()
