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
  return f"{CACHE_PREFIX}__{service_name}"