

tasti_engine = {
    'blob_mapping_table_name': 'blobs_mapping',
    'index_name': 'tasti',
}


'tasti_config': {
  'index_name': 'tasti',
  'nb_buckets': 1000,
  'percent_fpf': 0.75,
  'seed': 1234
},

Vector_database = {
  'vector_database_type': 'FAISS',
  'index_path': './',
  'WeaviateAuth': {
    'url': None,
    'api_key': None
  }
}
