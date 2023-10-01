from aidb.query.column_extractor import ColumnExtractor
from aidb.utils.logger import logger

import sqlglot.expressions as exp
from sqlglot import Tokenizer, Parser

from collections import defaultdict
from dataclasses import dataclass
from typing import Union, Any


def is_aqp_exp(node):
  return isinstance(node, exp.ErrorTarget) \
    or isinstance(node, exp.RecallTarget) \
    or isinstance(node, exp.PrecisionTarget) \
    or isinstance(node, exp.Confidence) \
    or isinstance(node, exp.Budget)

def _remove_aqp(node):
  if is_aqp_exp(node):
    return None
  return node


class Query(object):
  def __init__(self, sql):
    self.sql = sql
    self._tokens = Tokenizer().tokenize(sql)
    self._expression: exp.Expression = Parser().parse(self._tokens)[0]
    self.transform_sql_to_base()

  def transform_sql_to_base(self):
    if not hasattr(self, '_exp_no_aqp'):
      self._exp_no_aqp = self._expression.transform(_remove_aqp)
    if self._exp_no_aqp is None:
      raise Exception('SQL contains no non-AQP statements')
    return self._exp_no_aqp

  def get_sql_query_text(self):
    return self._expression.sql()

  # Get aggregation type
  def get_agg_type(self):
    if len(self._exp_no_aqp.args['expressions']) != 1:
      raise Exception('Multiple expressions found')
    select_exp = self._exp_no_aqp.args['expressions'][0]
    if isinstance(select_exp, exp.Avg):
      return exp.Avg
    elif isinstance(select_exp, exp.Count):
      return exp.Count
    elif isinstance(select_exp, exp.Sum):
      return exp.Sum
    else:
      logger.debug('Unsupported aggregation', select_exp)
      return None

  # Extract tables from query
  def get_tables_in_query(self):
    table_list = set()
    for node, _, _ in self._expression.walk():
      if isinstance(node, exp.Table):
        table_list.add(node.args["this"].args["this"])
    return table_list

  def get_aggregated_column(self, agg_type):
    """
    returns the column name that is aggregated in the query.
    for e.g. SELECT Avg(sentiment) from sentiments;
    will return sentiment
    """
    agg_exp_tree = self._expression.find(agg_type)
    for node, _, key in agg_exp_tree.walk():
      if isinstance(node, exp.Column):
        if isinstance(node.args['this'], exp.Identifier):
          return node.args['this'].args['this']
    return None

  def extract_columns(self):
    extractor = ColumnExtractor()
    return extractor.extract(self._exp_no_aqp)

  def is_approx_agg_query(self):
    return True if self.get_agg_type() else False

  def process_aggregate_query(self):
    self.tables_concerned = self.get_tables_in_query()
    self.columns_concerned = self.extract_columns()
    self.agg_type = self.get_agg_type()
    self.aggregated_column = self.get_aggregated_column(self.agg_type)