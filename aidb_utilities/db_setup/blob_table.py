from typing import List, Union

import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import MetaData
from sqlalchemy.sql import text

from aidb_utilities.blob_store.blob_store import Blob
from aidb.config.config_types import python_type_to_sqlalchemy_type
from aidb.utils.asyncio import asyncio_run
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE
from aidb.utils.db import create_sql_engine, infer_dialect


class BaseTablesSetup(object):

  def __init__(self, connection_uri):
    self._dialect = infer_dialect(connection_uri)
    self._sql_engine = create_sql_engine(connection_uri)


  async def _create_table(self, table_name, table_columns):
    '''
    creates the table in the database
    '''
    async with self._sql_engine.begin() as conn:
      metadata = MetaData(bind=conn)
      _ = sqlalchemy.Table(table_name, metadata, *[
        sqlalchemy.Column(c_name, c_type, primary_key=is_pk) for c_name, c_type, is_pk in table_columns
      ])
      await conn.run_sync(lambda conn: metadata.create_all(conn))


  async def _setup_blob_config_table(self, table_name, table_columns):
    '''
    setup the blob_key configuration table
    '''
    await self._create_table(BLOB_TABLE_NAMES_TABLE,
                             [('table_name', sqlalchemy.String(20), True), ('blob_key', sqlalchemy.String(20), True)])
    async with self._sql_engine.begin() as conn:
      for c_name, _, is_pk in table_columns:
        try:
          if is_pk:
            # Insert into blob metadata table
            await conn.execute(
              text(f'INSERT INTO {BLOB_TABLE_NAMES_TABLE} VALUES (:table_name, :blob_key)')
              .bindparams(table_name=table_name, blob_key=c_name)
            )
        except sqlalchemy.exc.IntegrityError:
          print(f"Skipping: Blob config table already have {table_name} and {c_name}")


  async def _insert_data_in_table(self, table_name: str, data: pd.DataFrame):
    '''
    inserts rows in the table
    '''
    async with self._sql_engine.begin() as conn:
      try:
        await conn.run_sync(lambda conn: data.to_sql(table_name, conn, if_exists='append', index=False))
      except sqlalchemy.exc.IntegrityError:
        print(f"Skipping: Blob table is already populated with the blobs")


  def insert_blob_meta_data(self, table_name, blob_data: Union[pd.DataFrame, List[Blob]], primary_key_cols: List[str]):
    '''
    creates the blob table and the blob key configuration table
    inserts the blob data in the blob table
    '''
    assert len(primary_key_cols) > 0, 'Primary key should be specified'
    if not isinstance(blob_data, pd.DataFrame):
      blob_data = pd.DataFrame([b.to_dict() for b in blob_data])
    assert blob_data.shape[0] > 0, 'No blobs to insert in the blob table'
    table_columns = []
    for column in blob_data.columns:
      dtype = python_type_to_sqlalchemy_type(blob_data[column].dtype)
      if dtype == sqlalchemy.String:
        c_type = sqlalchemy.Text()
      else:
        c_type = dtype
      table_columns.append((column, c_type, column in primary_key_cols))
    asyncio_run(self._create_table(table_name, table_columns))
    asyncio_run(self._setup_blob_config_table(table_name, table_columns))
    asyncio_run(self._insert_data_in_table(table_name, blob_data))
