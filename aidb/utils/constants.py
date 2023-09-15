import hashlib
from typing import List


CACHE_PREFIX = '__cache'
CONFIG_PREFIX = '__config'

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
