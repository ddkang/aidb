import unittest
from aidb.query.query import Query

# Valid
valid_avg_sql = '''
SELECT AVG(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95;
'''

# Invalid
unsupported_agg_aqp_sql = '''
SELECT MAX(bar)
FROM foo
ERROR_TARGET 1%
CONFIDENCE 95;
'''

agg_no_et_sql = '''
SELECT AVG(bar)
FROM foo
CONFIDENCE 95;
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
    self.assertEqual(query.validate_aqp_or_throw(), True)

  def test_invalid_unsupported_agg_aqp(self):
    query = Query(unsupported_agg_aqp_sql, None)
    with self.assertRaises(Exception):
      query.validate_aqp_or_throw()

  def test_invalid_agg_no_et(self):
    query = Query(agg_no_et_sql, None)
    with self.assertRaises(Exception):
      query.validate_aqp_or_throw()

  def test_invalid_agg_no_conf(self):
    query = Query(agg_no_conf_sql, None)
    with self.assertRaises(Exception):
      query.validate_aqp_or_throw()

  def test_invalid_not_select_sql(self):
    query = Query(not_select_sql, None)
    with self.assertRaises(Exception):
      query.validate_aqp_or_throw()


if __name__ == '__main__':
  unittest.main()