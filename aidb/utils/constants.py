CACHE_PREFIX = '__cache'
CONFIG_PREFIX = '__config'

'''
The schema of the blob metadata table is as follows:
  table_name: str
  blob_key: str
'''
BLOB_TABLE_NAMES_TABLE = CONFIG_PREFIX + '_blob_tables'

def cache_table_name_from_inputs(columns: List[str]):
  cache_table_postfix = ''
  for idx, column in enumerate(columns):
    # Special characters are not allowed in table names
    cache_table_postfix += f'__{idx}_{column}__'
  return CACHE_PREFIX + cache_table_postfix