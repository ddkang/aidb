CACHE_PREFIX = '__cache'
CONFIG_PREFIX = '__config'

'''
The schema of the blob metadata table is as follows:
  table_name: str
  blob_key: str
'''
BLOB_TABLE_NAMES_TABLE = CONFIG_PREFIX + '_blob_tables'