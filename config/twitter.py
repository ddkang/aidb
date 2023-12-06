from aidb.config.config_types import AIDBListType
from aidb.inference.http_inference_service import HTTPInferenceService

DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb_test_twitter.sqlite'

entity_service = HTTPInferenceService(
  "entities00",
  False,
  url=f'http://127.0.0.1:8000/entities00',
  headers={'Content-Type': 'application/json'},
  columns_to_input_keys=['blobs00.tweet_id'],
  response_keys_to_columns=[('entities00.type', AIDBListType()), ('entities00.entity', AIDBListType()),
                            ('entities00.tweet_id', AIDBListType()), ('entities00.entity_id', AIDBListType())]
)

hate_service = HTTPInferenceService(
  "hate01",
  False,
  url=f'http://127.0.0.1:8000/hate01',
  headers={'Content-Type': 'application/json'},
  columns_to_input_keys=['blobs00.tweet_id'],
  response_keys_to_columns=[('hate01.tweet_id', AIDBListType()), ('hate01.ishate', AIDBListType())]
)

sentiment_service = HTTPInferenceService(
  "sentiment02",
  False,
  url=f'http://127.0.0.1:8000/sentiment02',
  headers={'Content-Type': 'application/json'},
  columns_to_input_keys=['blobs00.tweet_id'],
  response_keys_to_columns=[('sentiment02.tweet_id', AIDBListType()), ('sentiment02.sentiment', AIDBListType())]
)

topic_service = HTTPInferenceService(
  "topic03",
  False,
  url=f'http://127.0.0.1:8000/topic03',
  headers={'Content-Type': 'application/json'},
  columns_to_input_keys=['blobs00.tweet_id'],
  response_keys_to_columns=[('topic03.tweet_id', AIDBListType()), ('topic03.topic', AIDBListType())]
)

inference_engines = [
  {
    "service": entity_service,
    "input_col": ("blobs00.tweet_id",),
    "output_col": ('entities00.type', 'entities00.entity', 'entities00.tweet_id', 'entities00.entity_id')
  },
  {
    "service": hate_service,
    "input_col": ("blobs00.tweet_id",),
    "output_col": ('hate01.tweet_id', 'hate01.ishate')
  },
  {
    "service": sentiment_service,
    "input_col": ("blobs00.tweet_id",),
    "output_col": ('sentiment02.tweet_id', 'sentiment02.sentiment')
  },
  {
    "service": topic_service,
    "input_col": ("blobs00.tweet_id",),
    "output_col": ('topic03.tweet_id', 'topic03.topic')
  }
]

blobs_csv_file = "tests/data/tweets.csv"
blob_table_name = "blobs00"
blobs_keys_columns = ["tweet_id"]

"""
dictionary of table names to list of columns
"""

tables = {
  "entities00": [
    {"name": "tweet_id", "is_primary_key": True, "refers_to": ("blobs00", "tweet_id"), "dtype": int},
    {"name": "entity_id", "is_primary_key": True, "dtype": int},
    {"name": "entity", "dtype": str},
    {"name": "type", "dtype": str}],
  "hate01": [
    {"name": "tweet_id", "is_primary_key": True, "refers_to": ("blobs00", "tweet_id"), "dtype": int},
    {"name": "ishate", "dtype": int}],
  "sentiment02": [
    {"name": "tweet_id", "is_primary_key": True, "refers_to": ("blobs00", "tweet_id"), "dtype": int},
    {"name": "sentiment", "dtype": int}],
  "topic03": [
    {"name": "tweet_id", "is_primary_key": True, "refers_to": ("blobs00", "tweet_id"), "dtype": int},
    {"name": "topic", "dtype": str}],
}

INITIALIZE_TASTI = False
