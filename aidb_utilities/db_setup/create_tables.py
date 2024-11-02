from typing import Dict

import sqlalchemy
import sqlalchemy.ext.asyncio
from sqlalchemy import MetaData
from sqlalchemy.schema import ForeignKeyConstraint

from aidb.config.config_types import python_type_to_sqlalchemy_type
from aidb.utils.db import create_sql_engine


async def create_output_tables(db_config: Dict[str, str], output_tables: Dict):
  db_uri = f"{db_config['url']}/{db_config['name']}"
  engine = create_sql_engine(db_uri)
  async with engine.begin() as conn:
    metadata = MetaData(bind=conn)
    await conn.run_sync(metadata.reflect)
    # Create tables
    existing_tables = metadata.tables
    for output_table in output_tables:
      table_name = output_table['name']
      columns = output_table['columns']
      if table_name in existing_tables:
        print(f"Skipping: Table {table_name} already exists")
        continue
      columns_info = []
      fk_constraints = {}
      for column in columns:
        for col_name, col_details in column.items():
          dtype = python_type_to_sqlalchemy_type(col_details["dtype"])
          if dtype == sqlalchemy.String:
            dtype = sqlalchemy.Text()
          is_primary_key = col_details.get('is_primary_key', False)
          columns_info.append(sqlalchemy.Column(col_name, dtype, primary_key=is_primary_key))

          if "refers_to" in col_details:
            fk_ref_table_name = col_details["refers_to"].split('.')[0]
            if fk_ref_table_name not in fk_constraints:
              fk_constraints[fk_ref_table_name] = {'cols': [], 'cols_refs': []}
            # both tables will have same column name
            fk_constraints[fk_ref_table_name]['cols'].append(col_name)
            fk_constraints[fk_ref_table_name]['cols_refs'].append(col_details["refers_to"])

      multi_table_fk_constraints = []
      for tbl, fk_cons in fk_constraints.items():
        multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))

      _ = sqlalchemy.Table(table_name, metadata, *columns_info, *multi_table_fk_constraints)

    await conn.run_sync(lambda conn: metadata.create_all(conn))
