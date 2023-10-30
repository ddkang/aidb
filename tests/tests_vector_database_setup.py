import numpy as np
import os
import unittest

from aidb_utilities.vector_database_setup.vector_database_setup import VectorDatabaseSetup
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateAuth, WeaviateVectorDatabase
from tests.tests_data_store import AidbDataStoreTests

from unittest import IsolatedAsyncioTestCase

DB_URL = 'sqlite+aiosqlite:///aidb_datastore.sqlite'


def clean_vector_database():
  if os.path.exists('tasti.index'):
    os.remove('tasti.index')
  if os.path.exists('chroma.sqlite3'):
    os.remove('chroma.sqlite3')


def test_equality(value):
  np.random.seed(1234)
  embeddings = np.random.rand(3, 128)
  embeddings = (embeddings * 100).astype(int)
  value = (value * 100).astype(int)
  assert np.array_equal(embeddings, value)


blob_table_name = 'blob00'
blob_mapping_table_name = 'blob_mapping_00'
index_name = 'tasti'

class AidbVectorDatabaseSetupTests(IsolatedAsyncioTestCase):

  async def test_faiss_set_up(self):
    clean_vector_database()
    vd_type = 'FAISS'
    auth = './'

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = FaissVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(3)), reload=True)
    test_equality(value)


  async def test_chroma_set_up(self):
    clean_vector_database()
    vd_type = 'chroma'
    auth = './'

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = ChromaVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(3)), reload=True)
    test_equality(value)


  @unittest.skip("Skip in case of absence of Weaviate credentials")
  async def test_weaviate_set_up(self):
    clean_vector_database()
    vd_type = 'weaviate'
    url = ''
    api_key = os.environ.get('WEAVIATE_API_KEY')
    auth = WeaviateAuth(url=url, api_key=api_key)

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = WeaviateVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(3)), reload=True)
    test_equality(value)


if __name__ == '__main__':
  unittest.main()