import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import text
from typing import List

from aidb.config.config_types import python_type_to_sqlalchemy_type
from aidb.utils.asyncio import asyncio_run
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE
from aidb.utils.db import create_sql_engine, infer_dialect


class BaseTablesSetup(object):

  def __init__(self, connection_uri):
    self._dialect = infer_dialect(connection_uri)
    self._sql_engine = create_sql_engine(connection_uri)

  async def _create_table(self, table_name, table_columns):
    async with self._sql_engine.begin() as conn:
      metadata = MetaData(bind=conn)
      _ = sqlalchemy.Table(table_name, metadata, *[
        sqlalchemy.Column(c_name, c_type, primary_key=is_pk) for c_name, c_type, is_pk in table_columns
      ])
      await conn.run_sync(lambda conn: metadata.create_all(conn))

  async def _setup_blob_config_table(self, table_name, table_columns):
    def create_blob_metadata_table(conn):
      Base = declarative_base()

      class BlobTables(Base):
        __tablename__ = BLOB_TABLE_NAMES_TABLE
        table_name = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
        blob_key = sqlalchemy.Column(sqlalchemy.String, primary_key=True)

      Base.metadata.create_all(conn)

    async with self._sql_engine.begin() as conn:
      await conn.run_sync(create_blob_metadata_table)
      for c_name, _, is_pk in table_columns:
        if is_pk:
          # Insert into blob metadata table
          await conn.execute(
            text(f'INSERT INTO {BLOB_TABLE_NAMES_TABLE} VALUES (:table_name, :blob_key)')
            .bindparams(table_name=table_name, blob_key=c_name)
          )

  async def _insert_data_in_table(self, table_name: str, data: pd.DataFrame):
    async with self._sql_engine.begin() as conn:
      await conn.run_sync(lambda conn: data.to_sql(table_name, conn, if_exists='append', index=False))

  def insert_data(self, table_name, blob_data: pd.DataFrame, primary_key_cols: List[str]):
    table_columns = []
    for column in blob_data.columns:
      dtype = python_type_to_sqlalchemy_type(blob_data[column].dtype)
      if dtype == sqlalchemy.String:
        # TODO: VAR CHAR lenth should be based on the number of characters
        c_type = sqlalchemy.String(20)  # 20 characters long
      else:
        c_type = dtype
      table_columns.append((column, c_type, column in primary_key_cols))
    asyncio_run(self._create_table(table_name, table_columns))
    asyncio_run(self._setup_blob_config_table(table_name, table_columns))
    asyncio_run(self._insert_data_in_table(table_name, blob_data))
