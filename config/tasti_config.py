import os

from aidb.vector_database.weaviate_vector_database import WeaviateAuth


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
  'nb_buckets': 1000,
  'percent_fpf': 0.75,
  'seed': 1234,
}


