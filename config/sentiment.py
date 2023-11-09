import os

from aidb.config.config_types import AIDBListType
from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
from aidb.vector_database.weaviate_vector_database import WeaviateAuth


DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb_test_amazon.sqlite'

sentiment_inference_service = HuggingFaceNLP(
  name="sentiment_classification",
  token=None, # leave it None if you want AIDB to read token from env variable HF_API_KEY. Otherwise replace None with your own token in str.
  columns_to_input_keys=['inputs'],
  response_keys_to_columns=[(AIDBListType(), AIDBListType(), 'label'),
                            (AIDBListType(), AIDBListType(), 'score')],
  input_columns_types=[str],
  output_columns_types=[str, float],
  model="LiYuan/amazon-review-sentiment-analysis",
  default_args={("options", "wait_for_model"): True})

inference_engines = [
  {
    "service": sentiment_inference_service,
    "input_col": ("blobs00.review", "blobs00.review_id"),
    "output_col": ("sentiment.label", "sentiment.score", "sentiment.review_id")
  }
]

blobs_csv_file = "tests/data/amazon_reviews.csv"
blob_table_name = "blobs00"
blobs_keys_columns = ["review_id"]

"""
dictionary of table names to list of columns
"""

tables = {"sentiment": [
  {"name": "review_id", "is_primary_key": True, "refers_to": ("blobs00", "review_id"), "dtype": int},
  {"name": "label", "is_primary_key": True, "dtype": str},
  {"name": "score", "dtype": float}]}


INITIALIZE_TASTI = False

# The configuration below is necessary when initializing TASTI for the first time.
# required for Weaviate, not for FAISS or Chroma
url = ''
api_key = os.environ.get('WEAVIATE_API_KEY')
weaviate_auth = WeaviateAuth(url=url, api_key=api_key)

# Optional, use to choose part of vector id for analysis
vector_id_csv = None

vector_database = {
  # Currently support FAISS, Chroma and weaviate
  'vector_database_type': 'FAISS',
  'auth': {
    # path for FAISS and Chroma, weaviate_auth for weaviate, only one needed.
    'path': './',
    # 'weaviate_auth': weaviate_auth
  }
}

tasti_engine = {
  'index_name': 'tasti',
  # below are optional config
  'nb_buckets': 100,
  'percent_fpf': 0.75,
  'seed': 1234,
}
