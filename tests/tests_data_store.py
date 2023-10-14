import pandas as pd
import unittest
import os

from dataclasses import asdict

from aidb.blob_store.local_storage import LocalImageBlobStore
from aidb.db_setup.blob_table import BaseTablesSetup


class LocalDataStoreTests(unittest.TestCase):
  def test_positive(self):
    dirname = os.path.dirname(__file__)
    data_dir = os.path.join(dirname, 'data/data_store/')
    local_image_store = LocalImageBlobStore("/home/akash/Pictures")
    image_blobs = local_image_store.get_blobs()
    base_table_setup = BaseTablesSetup("sqlite+aiosqlite:///aidb_datastore.sqlite")
    base_table_setup.insert_data("blob00", image_blobs, ["blob_id"])


if __name__ == '__main__':
  unittest.main()
