import unittest
import os
import sqlalchemy
import sqlalchemy.ext.asyncio
from aidb.blob_store.local_storage import LocalImageBlobStore
from aidb.db_setup.blob_table import BaseTablesSetup
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

DB_URL = "sqlite+aiosqlite:///aidb_datastore.sqlite"


class LocalDataStoreTests(IsolatedAsyncioTestCase):
  async def test_positive(self):
    if os.path.exists("aidb_datastore.sqlite"):
      os.remove("aidb_datastore.sqlite")
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/data_store/')
    local_image_store = LocalImageBlobStore("/home/akash/Pictures")
    image_blobs = local_image_store.get_blobs()
    base_table_setup = BaseTablesSetup(DB_URL)
    blob_table = "blob00"
    base_table_setup.insert_data(blob_table, image_blobs, ["blob_id"])
    engine = sqlalchemy.ext.asyncio.create_async_engine(DB_URL)
    async with engine.begin() as conn:
      result = await conn.execute(text(f"SELECT * FROM {blob_table}"))
      total_blobs = result.fetchall()
      result = await conn.execute(text(f"SELECT * FROM {BLOB_TABLE_NAMES_TABLE}"))
      total_blob_keys = result.fetchall()
    assert len(total_blobs) == len(image_blobs)
    assert len(total_blob_keys) == 1


if __name__ == '__main__':
  unittest.main()
