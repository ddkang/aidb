import numpy as np
import os
import sqlalchemy
import sqlalchemy.ext.asyncio
import unittest

from aidb_utilities.blob_store.local_storage import LocalImageBlobStore
from aidb_utilities.db_setup.blob_table import BaseTablesSetup
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE
from aidb_utilities.vector_database_setup.vector_database_setup import VectorDatabaseSetup
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateAuth, WeaviateVectorDatabase

from sqlalchemy.sql import text
from unittest import IsolatedAsyncioTestCase

DB_URL = 'sqlite+aiosqlite:///aidb_datastore.sqlite'


async def setup_blob_tables(blob_table_name):
  dirname = os.path.dirname(__file__)
  data_dir = os.path.join(dirname, 'data/image_data_store/')
  local_image_store = LocalImageBlobStore(data_dir)
  image_blobs = local_image_store.get_blobs()

  base_table_setup = BaseTablesSetup(DB_URL)
  base_table_setup.insert_blob_meta_data(blob_table_name, image_blobs, ['blob_id'])
  engine = sqlalchemy.ext.asyncio.create_async_engine(DB_URL)
  async with engine.begin() as conn:
    result = await conn.execute(text(f'SELECT * FROM {blob_table_name}'))
    total_blobs = result.fetchall()
    result = await conn.execute(text(f'SELECT * FROM {BLOB_TABLE_NAMES_TABLE}'))
    total_blob_keys = result.fetchall()
  assert len(total_blobs) == len(image_blobs)
  assert len(total_blob_keys) == 1


def clean_resources():
  if os.path.exists('aidb_datastore.sqlite'):
    os.remove('aidb_datastore.sqlite')
  if os.path.exists('tasti.index'):
    os.remove('tasti.index')
  if os.path.exists('chroma.sqlite3'):
    os.remove('chroma.sqlite3')


def test_equality(value):
  np.random.seed(1234)
  embeddings = np.random.rand(1000, 128)
  embeddings = (embeddings * 100).astype(int)
  value = (value * 100).astype(int)
  assert np.array_equal(embeddings, value)


class AidbDataStoreTests(IsolatedAsyncioTestCase):

  async def test_faiss_set_up(self):
    clean_resources()
    blob_table_name = 'blob00'
    blob_mapping_table_name = 'blob_mapping_00'
    vd_type = 'FAISS'
    index_name = 'tasti'
    auth = './'
    await setup_blob_tables(blob_table_name)

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = FaissVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(1000)), reload=True)
    test_equality(value)


  async def test_chroma_set_up(self):
    clean_resources()
    blob_table_name = 'blob00'
    blob_mapping_table_name = 'blob_mapping_00'
    vd_type = 'chroma'
    index_name = 'tasti'
    auth = './'
    await setup_blob_tables(blob_table_name)

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = ChromaVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(1000)), reload=True)
    test_equality(value)


  @unittest.skip("Skip in case of absence of Weaviate credentials")
  async def test_weaviate_set_up(self):
    clean_resources()
    blob_table_name = 'blob00'
    blob_mapping_table_name = 'blob_mapping_00'
    vd_type = 'weaviate'
    index_name = 'tasti'
    url = ''
    api_key = os.environ.get('WEAVIATE_API_KEY')
    auth = WeaviateAuth(url=url, api_key=api_key)

    await setup_blob_tables(blob_table_name)

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

    existing_vector_database = WeaviateVectorDatabase(auth)
    value = existing_vector_database.get_embeddings_by_id(index_name, ids=np.array(range(1000)), reload=True)
    test_equality(value)


if __name__ == '__main__':
  unittest.main()