import hashlib
from typing import List


CACHE_PREFIX = '__cache'
CONFIG_PREFIX = '__config'
REP_PREFIX = '__rep'
TOPK_PREFIX = '__TOPK'
'''
The schema of the blob metadata table is as follows:
  table_name: str
  blob_key: str
'''
BLOB_TABLE_NAMES_TABLE = CONFIG_PREFIX + '_blob_tables'

def cache_table_name_from_inputs(service_name: str, columns: List[str]):
  cache_table_postfix = ''
  for idx, column in enumerate(columns):
    # Special characters are not allowed in table names
    column = column.replace('.', '__')
    cache_table_postfix += f'__{idx}_{column}__'
  # limit to 10 characters to avoid long names in the table names
  hash_length = 10
  column_input_hash = hashlib.sha1(cache_table_postfix.encode()).hexdigest()[:hash_length]
  return f"{CACHE_PREFIX}__{service_name}__{column_input_hash}"


def table_name_for_rep_and_topk(blob_tables: List[str]):
  blob_tables.sort()
  blob_table_postfix = ''
  for idx, blob_table in enumerate(blob_tables):
    # Special characters are not allowed in table names
    column = column.replace('.', '__')
    blob_table_postfix += f'__{blob_table}__'
  # limit to 10 characters to avoid long names in the table names
  hash_length = 10
  table_input_hash = hashlib.sha1(blob_table_postfix.encode()).hexdigest()[:hash_length]
  return f"{REP_PREFIX}__{table_input_hash}", f"{TOPK_PREFIX}__{table_input_hash}"