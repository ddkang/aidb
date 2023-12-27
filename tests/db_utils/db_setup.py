import glob
import os
import networkx as nx
import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import MetaData
from sqlalchemy.schema import ForeignKeyConstraint
from sqlalchemy.ext.declarative import declarative_base

from aidb.config.config_types import python_type_to_sqlalchemy_type
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE, table_name_for_rep_and_topk_and_blob_mapping
from aidb.utils.logger import logger
from sqlalchemy.sql import text
from dataclasses import dataclass
from typing import Optional


@dataclass
class ColumnInfo:
  name: str
  is_primary_key: bool
  refers_to: Optional[tuple]  # (table, column)
  dtype = None


def extract_column_info(table_name, column_str) -> ColumnInfo:
  pk = False
  if column_str.startswith("pk_"):
    pk = True
    column_str = column_str[3:]  # get rid of pk_ prefix
  t, c = column_str.split('.')
  fk = None
  if t != table_name:
    fk = (t, c)
  return ColumnInfo(c, pk, fk)


async def create_db(db_url: str, db_name: str):
  dialect = db_url.split("+")[0]
  if dialect == "postgresql" or dialect == "mysql":
    engine = sqlalchemy.ext.asyncio.create_async_engine(db_url, isolation_level='AUTOCOMMIT')
    try:
      async with engine.begin() as conn:
        await conn.execute(text(f"CREATE DATABASE {db_name}"))
    except sqlalchemy.exc.ProgrammingError:
      logger.warning(f'Database {db_name} Already exists')
    finally:
      await engine.dispose()
  elif dialect == "sqlite":
    # sqlite auto creates, do nothing
    pass
  else:
    raise NotImplementedError
  return


async def drop_all_tables(conn):
  metadata = MetaData(bind=conn)
  # Reflect the database to get all table names
  await conn.run_sync(metadata.reflect)
  await conn.run_sync(metadata.drop_all)
  return


async def setup_db(db_url: str, db_name: str, data_dir: str):
  gt_dir = f'{data_dir}/ground_truth'
  gt_csv_fnames = glob.glob(f'{gt_dir}/*.csv')
  gt_csv_fnames.sort()

  db_uri = f'{db_url}/{db_name}'
  # Connect to db
  engine = sqlalchemy.ext.asyncio.create_async_engine(db_uri)
  try:
    async with engine.begin() as conn:
      await drop_all_tables(conn)
      metadata = MetaData(bind=conn)
      # Create tables
      for csv_fname in gt_csv_fnames:
        base_fname = os.path.basename(csv_fname)
        table_name = base_fname.split('.')[0]
        df = pd.read_csv(csv_fname)
        columns_info = []
        fk_constraints = {}
        for column in df.columns:
          column_info = extract_column_info(table_name, column)
          dtype = python_type_to_sqlalchemy_type(df[column].dtype)
          if dtype == sqlalchemy.String:
            # TODO: VAR CHAR lenth should be based on the number of characters
            if column_info.is_primary_key or column_info.refers_to:
              column_info.dtype = sqlalchemy.VARCHAR(256)
            else:
              column_info.dtype = sqlalchemy.TEXT
          else:
            column_info.dtype = dtype
          columns_info.append(column_info)
          df.rename(columns={column: column_info.name}, inplace=True)

          if column_info.refers_to is not None:
            fk_ref_table_name = column_info.refers_to[0]
            if fk_ref_table_name not in fk_constraints:
              fk_constraints[fk_ref_table_name] = {'cols': [], 'cols_refs': []}
            # both tables will have same column name
            fk_constraints[fk_ref_table_name]['cols'].append(column_info.name)
            fk_constraints[fk_ref_table_name]['cols_refs'].append(
              f"{column_info.refers_to[0]}.{column_info.refers_to[1]}")

        multi_table_fk_constraints = []
        for tbl, fk_cons in fk_constraints.items():
          multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))

        _ = sqlalchemy.Table(
            table_name,
            metadata,
            *[sqlalchemy.Column(c_info.name, c_info.dtype, primary_key=c_info.is_primary_key, autoincrement=False)
                for c_info in columns_info],
            *multi_table_fk_constraints
        )

      await conn.run_sync(lambda conn: metadata.create_all(conn))
  finally:
    await engine.dispose()
  return engine


async def insert_data_in_tables(conn, data_dir: str, only_blob_data: bool):
  def get_insertion_order(conn, gt_csv_files):
    metadata = MetaData()
    metadata.reflect(conn)
    table_graph = nx.DiGraph()
    for table in metadata.sorted_tables:
      for fk_col in table.foreign_keys:
        parent_table = str(fk_col.column).split('.')[0]
        table_graph.add_edge(parent_table, table.name)
    table_order = nx.topological_sort(table_graph)
    ordered_csv_files = []
    for table_name in table_order:
      csv_file_name = f"{table_name}.csv"
      for f in gt_csv_files:
        if csv_file_name in f:
          ordered_csv_files.append(f)
          break
    return ordered_csv_files

  gt_dir = f'{data_dir}/ground_truth'
  gt_csv_fnames = glob.glob(f'{gt_dir}/*.csv')

  gt_csv_fnames = await conn.run_sync(get_insertion_order, gt_csv_fnames, )
  # Create tables
  for csv_fname in gt_csv_fnames:
    base_fname = os.path.basename(csv_fname)
    table_name = base_fname.split('.')[0]

    if only_blob_data and not table_name.startswith('blobs') and not table_name.startswith('mapping_'):
      continue
    df = pd.read_csv(csv_fname)
    for column in df.columns:
      column_info = extract_column_info(table_name, column)
      df.rename(columns={column: column_info.name}, inplace=True)

    if table_name.startswith('mapping'):
      _, _, blob_mapping_table_name = table_name_for_rep_and_topk_and_blob_mapping(['blobs_00'])
      table_name = blob_mapping_table_name

    await conn.run_sync(lambda conn: df.to_sql(table_name, conn, if_exists='append', index=False))


async def clear_all_tables(conn):
  def tmp(conn):
    metadata = MetaData()
    metadata.reflect(conn)
    for table in metadata.sorted_tables:
      if table.name.startswith('blobs'):
        continue
      conn.execute(table.delete())
  await conn.run_sync(tmp)


async def setup_config_tables(conn):
  def create_blob_metadata_table(conn):
    Base = declarative_base()

    class BlobTables(Base):
      __tablename__ = BLOB_TABLE_NAMES_TABLE
      table_name = sqlalchemy.Column(sqlalchemy.VARCHAR(256), primary_key=True)
      blob_key = sqlalchemy.Column(sqlalchemy.VARCHAR(256), primary_key=True)

    Base.metadata.create_all(conn)


  def get_blob_table_names_and_columns(conn):
    metadata = MetaData()
    metadata.reflect(conn)
    table_names = [table.name for table in metadata.sorted_tables if table.name.startswith('blob')]
    # Get the columns for each table
    table_names_and_columns = {}
    for table_name in table_names:
      table = metadata.tables[table_name]
      table_names_and_columns[table_name] = [column.name for column in table.columns if column.primary_key]
    return table_names, table_names_and_columns

  await conn.run_sync(create_blob_metadata_table)
  blob_table_names, columns = await conn.run_sync(get_blob_table_names_and_columns)
  for table_name in blob_table_names:
    for column in columns[table_name]:
      # Insert into blob metadata table
      await conn.execute(
        text(f'INSERT INTO {BLOB_TABLE_NAMES_TABLE} VALUES (:table_name, :blob_key)')
        .bindparams(table_name=table_name, blob_key=column)
      )
