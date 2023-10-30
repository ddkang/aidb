import unittest
import os
import sqlalchemy
import sqlalchemy.ext.asyncio
from aidb_utilities.blob_store.local_storage import LocalImageBlobStore
from aidb_utilities.db_setup.blob_table import BaseTablesSetup
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE
from aidb_utilities.vector_database_setup.vector_database_setup import VectorDatabaseSetup

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

DB_URL = 'sqlite+aiosqlite:///aidb_datastore.sqlite'


async def setup_blob_tables(input_blobs, blob_table_name):
  base_table_setup = BaseTablesSetup(DB_URL)
  base_table_setup.insert_blob_meta_data(blob_table_name, input_blobs, ['blob_id'])
  engine = sqlalchemy.ext.asyncio.create_async_engine(DB_URL)
  async with engine.begin() as conn:
    result = await conn.execute(text(f'SELECT * FROM {blob_table_name}'))
    total_blobs = result.fetchall()
    result = await conn.execute(text(f'SELECT * FROM {BLOB_TABLE_NAMES_TABLE}'))
    total_blob_keys = result.fetchall()
  assert len(total_blobs) == len(input_blobs)
  assert len(total_blob_keys) == 1


def clean_resources():
  if os.path.exists('aidb_datastore.sqlite'):
    os.remove('aidb_datastore.sqlite')


class AidbDataStoreTests(IsolatedAsyncioTestCase):

  async def test_vector_database_setup(self):
    clean_resources()
    blob_table_name = 'blob00'
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/image_data_store/')
    local_image_store = LocalImageBlobStore(data_dir)
    image_blobs = local_image_store.get_blobs()
    await setup_blob_tables(image_blobs, blob_table_name)

    blob_mapping_table_name = 'blob_mapping_00'
    vd_type = 'FAISS'
    index_name = 'tasti'
    auth = './'

    vector_database = VectorDatabaseSetup(DB_URL, blob_table_name, blob_mapping_table_name, vd_type, index_name, auth)
    await vector_database.setup()

if __name__ == '__main__':
  unittest.main()