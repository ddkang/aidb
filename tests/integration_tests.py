import asyncio
import glob
import os

import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
import uvicorn
from fastapi import FastAPI, Request
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

from aidb.config.config_types import (InferenceBinding,
                                      python_type_to_sqlalchemy_type)
from aidb.engine import Engine
from aidb.inference.http_inference_service import HTTPInferenceService
from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE


async def setup_db(data_dir: str, db_fname: str):
  gt_dir = f'{data_dir}/ground_truth'
  gt_csv_fnames = glob.glob(f'{gt_dir}/*.csv')
  gt_csv_fnames.sort()

  try:
    os.remove(db_fname)
  except:
    pass
  gt_db_uri = f'sqlite+aiosqlite:///{db_fname}'

  # Connect to db
  engine = sqlalchemy.ext.asyncio.create_async_engine(gt_db_uri)
  async with engine.begin() as conn:
    # Create tables
    for csv_fname in gt_csv_fnames:
      base_fname = os.path.basename(csv_fname)
      table_name = base_fname.split('.')[0]
      df = pd.read_csv(csv_fname)
      # TODO: need to assign foreign keys based on the table names
      for column in df.columns:
        column_name = column.split('.')[-1]
        df.rename(columns={column: column_name}, inplace=True)

      # Need to create the table with primary keys if it's a blob table
      if table_name.startswith('blobs'):
        # Get the schema from the df
        schema = {}
        for column in df.columns:
          column_name = column.split('.')[-1]
          schema[column_name] = python_type_to_sqlalchemy_type(df[column].dtype)
        # Create the table
        metadata = MetaData()
        _ = sqlalchemy.Table(table_name, metadata, *[
          sqlalchemy.Column(column_name, column_type, primary_key=True) for column_name, column_type in schema.items()
        ])
        await conn.run_sync(lambda conn: metadata.create_all(conn))

      await conn.run_sync(lambda conn: df.to_sql(table_name, conn, if_exists='append', index=False))
      print(df.head())
      print(table_name)

  return engine


async def clear_all_tables(engine):
  def tmp(conn):
    metadata = MetaData()
    metadata.reflect(conn)
    for table in metadata.sorted_tables:
      if table.name.startswith('blobs'):
        continue
      conn.execute(table.delete())

  async with engine.begin() as conn:
    await conn.run_sync(tmp)


async def setup_config_tables(engine):
  def create_blob_metadata_table(conn):
    Base = declarative_base()

    class BlobTables(Base):
      __tablename__ = BLOB_TABLE_NAMES_TABLE
      table_name = sqlalchemy.Column(sqlalchemy.String, primary_key=True)
      blob_key = sqlalchemy.Column(sqlalchemy.String, primary_key=True)

    Base.metadata.create_all(conn)

  def get_blob_table_names_and_columns(conn):
    metadata = MetaData()
    metadata.reflect(conn)
    table_names = [table.name for table in metadata.sorted_tables if table.name.startswith('blob')]
    # Get the columns for each table
    table_names_and_columns = {}
    for table_name in table_names:
      table = metadata.tables[table_name]
      table_names_and_columns[table_name] = [column.name for column in table.columns]
    return table_names, table_names_and_columns

  async with engine.begin() as conn:
    await conn.run_sync(create_blob_metadata_table)
    blob_table_names, columns = await conn.run_sync(get_blob_table_names_and_columns)
    for table_name in blob_table_names:
      for column in columns[table_name]:
        # Insert into blob metadata table
        await conn.execute(
          sqlalchemy.text(f'INSERT INTO {BLOB_TABLE_NAMES_TABLE} VALUES (:table_name, :blob_key)')
          .bindparams(table_name=table_name, blob_key=column)
        )


# TODO: currently, it hangs if it runs in the same process as the testing code
def run_server(data_dir: str):
  app = FastAPI()

  inference_dir = f'{data_dir}/inference'
  inference_csv_fnames = glob.glob(f'{inference_dir}/*.csv')
  inference_csv_fnames.sort()

  # Create the inference services
  name_to_df = {}
  for csv_fname in inference_csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    df = pd.read_csv(csv_fname)
    name_to_df[service_name] = df
    print('Creating service', service_name)

    @app.post(f'/{service_name}')
    async def inference(inp: Request):
      service_name = inp.url.path.split('/')[-1]
      try:
        inp = await inp.json()
        # Get the rows where the input columns match
        df = name_to_df[service_name]
        tmp = df[df[df.columns[0]] == inp[0]]
        for column, value in zip(df.columns[1:], inp[1:]):
          tmp = tmp[tmp[column] == value]
        res = tmp.to_dict(orient='list')
        print(res)
        return res
      except Exception as e:
        print('Error', e)
        return []

  # config = Config(app=app, host="127.0.0.1", port=8000)
  # server = Server(config=config)
  uvicorn.run(app, host="127.0.0.1", port=8000)


def register_inference_services(engine: Engine, data_dir: str):
  csv_fnames = glob.glob(f'{data_dir}/inference/*.csv')
  csv_fnames.sort()  # TODO: sort by number
  for csv_fname in csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    service = HTTPInferenceService(
      service_name,
      False,
      url=f'http://127.0.0.1:8000/{service_name}',
      headers={'Content-Type': 'application/json'},
    )
    engine.register_inference_service(service)

    def fix_output_col_name(col):
      tmp = col.split('.')[1]
      return service_name + '.' + tmp

    df = pd.read_csv(csv_fname)
    input_columns = [col for col in df.columns if not col.startswith(service_name)]
    output_columns = [fix_output_col_name(col) for col in df.columns]
    engine.bind_inference_service(
      service_name,
      InferenceBinding(
        tuple(input_columns),
        tuple(output_columns),
      )
    )


async def main():
  data_dir = '/home/ubuntu/aidb/aidb-new/tests/data/jackson'
  queries = [
    (
      'full_scan',
      '''SELECT * FROM `objects00`;''',
      '''SELECT * FROM `objects00`;''',
    )
  ]

  # Set up the ground truth database
  gt_db_fname = 'aidb-gt.sqlite'
  gt_engine = await setup_db(data_dir, gt_db_fname)

  # Set up the aidb database
  aidb_db_fname = 'aidb-test.sqlite'
  tmp_engine = await setup_db(data_dir, aidb_db_fname)
  await clear_all_tables(tmp_engine)
  await setup_config_tables(tmp_engine)
  del tmp_engine

  # Connect to the aidb database
  engine = Engine(
    f'sqlite+aiosqlite:///{aidb_db_fname}',
    debug=False,
  )

  # Register the services
  register_inference_services(engine, data_dir)

  for query_type, aidb_query, exact_query in queries:
    print(f'Running query {exact_query} in ground truth database')
    # Run the query on the ground truth database
    async with gt_engine.begin() as conn:
      gt_res = await conn.execute(sqlalchemy.text(exact_query))
      gt_res = gt_res.fetchall()
    # Run the query on the aidb database
    print(f'Running query {aidb_query} in aidb database')
    aidb_res = engine.execute(aidb_query)
    # TODO: check that the results are the same
    print(gt_res[0])
    print(aidb_res[0])


if __name__ == '__main__':
  asyncio.run(main())
