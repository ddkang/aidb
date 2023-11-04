import sqlalchemy.ext.asyncio
from sqlalchemy.schema import ForeignKeyConstraint

import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import MetaData

from aidb.config.config_types import python_type_to_sqlalchemy_type
from aidb.utils.db import create_sql_engine


async def create_output_tables(db_url: str, db_name: str, tables_info):
  db_uri = f'{db_url}/{db_name}'
  engine = create_sql_engine(db_uri)
  async with engine.begin() as conn:
    metadata = MetaData(bind=conn)
    await conn.run_sync(metadata.reflect)
    # Create tables
    existing_tables = metadata.tables
    for table_name, columns in tables_info.items():
      if table_name in existing_tables:
        print(f"Skipping: Table {table_name} already exists")
        continue
      columns_info = []
      fk_constraints = {}
      for column in columns:
        dtype = python_type_to_sqlalchemy_type(column["dtype"])
        if dtype == sqlalchemy.String:
          dtype = sqlalchemy.Text()
        is_primary_key = "is_primary_key" in column
        columns_info.append(sqlalchemy.Column(column["name"], dtype, primary_key=is_primary_key))

        if "refers_to" in column:
          fk_ref_table_name = column["refers_to"][0]
          if fk_ref_table_name not in fk_constraints:
            fk_constraints[fk_ref_table_name] = {'cols': [], 'cols_refs': []}
          # both tables will have same column name
          fk_constraints[fk_ref_table_name]['cols'].append(column["name"])
          fk_constraints[fk_ref_table_name]['cols_refs'].append(f"{column['refers_to'][0]}.{column['refers_to'][1]}")

      multi_table_fk_constraints = []
      for tbl, fk_cons in fk_constraints.items():
        multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))

      _ = sqlalchemy.Table(table_name, metadata, *columns_info, *multi_table_fk_constraints)

    await conn.run_sync(lambda conn: metadata.create_all(conn))
