import unittest
import os
import sqlalchemy
import sqlalchemy.ext.asyncio
from aidb.blob_store.local_storage import LocalImageBlobStore, LocalDocumentBlobStore
from aidb.blob_store.aws_s3_storage import AwsS3ImageBlobStore
from aidb.db_setup.blob_table import BaseTablesSetup
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE

from unittest import IsolatedAsyncioTestCase
from sqlalchemy.sql import text

DB_URL = "sqlite+aiosqlite:///aidb_datastore.sqlite"


async def setup_blob_tables(image_blobs):
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


def clean_resources():
  if os.path.exists("aidb_datastore.sqlite"):
    os.remove("aidb_datastore.sqlite")


class AidbDataStoreTests(IsolatedAsyncioTestCase):

  async def test_local_image_storage_positive(self):
    clean_resources()
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/image_data_store/')
    local_image_store = LocalImageBlobStore(data_dir)
    image_blobs = local_image_store.get_blobs()
    await setup_blob_tables(image_blobs)

  async def test_local_document_storage_positive(self):
    clean_resources()
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/document_data_store/')
    local_document_store = LocalDocumentBlobStore(data_dir)
    document_blobs = local_document_store.get_blobs()
    await setup_blob_tables(document_blobs)

  async def test_aws_image_storage_positive(self):
    clean_resources()
    aws_image_store = AwsS3ImageBlobStore("bucket-name", "<your-aws-access-key>", "your-secret-key")
    image_blobs = aws_image_store.get_blobs()
    await setup_blob_tables(image_blobs)


if __name__ == '__main__':
  unittest.main()
