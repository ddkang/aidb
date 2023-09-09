import asyncio

import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap
from sqlalchemy import insert

from sqlalchemy.sql import text
from sqlalchemy import (Boolean, Column, Float, Integer, MetaData,
                        Table, Text)
from sqlalchemy.schema import ForeignKeyConstraint

from aidb.utils.logger import logger


class Database():
    def __init__(
            self,
            config,
            debug: bool = False
    ):
        self._config = config['configuration']
        self._connection_uri = self._config['db_url']
        self._debug = debug

        self._dialect = self._infer_dialect(self._connection_uri)
        self._sql_engine = self._create_sql_engine()
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._create_table())

    def __del__(self):
        self._loop.run_until_complete(self._sql_engine.dispose())
        self._loop.close()

    # ---------------------
    # Setup
    # ---------------------
    def _infer_dialect(self, connection_uri: str):
        dialect = connection_uri.split(':')[0]
        if '+' in dialect:
            dialect = dialect.split('+')[0]

        supported_dialects = [
            'mysql',
            'postgresql',
            'sqlite',
        ]

        if dialect not in supported_dialects:
            logger.warning(f'Unsupported dialect: {dialect}. Defaulting to mysql')
            dialect = 'mysql'

        return dialect

    def _create_sql_engine(self):
        logger.info(f'Creating SQL engine for {self._dialect}')
        if self._dialect == 'mysql':
            kwargs = {
                'echo': self._debug,
                'max_overflow': -1,
            }
        else:
            kwargs = {}

        engine = sqlalchemy.ext.asyncio.create_async_engine(
            self._connection_uri,
            **kwargs,
        )

        return engine

    def create_table_sql(self):
        type_str_to_sql_type = {
            'int': Integer,
            'float': Float,
            'bool': Boolean,
            'str': Text,  # TODO: required for mysql?
        }
        self._metadata = MetaData()
        query_list = []
        tables = self._config['tables']
        for table_name in tables:
            table = tables[table_name]
            columns = []
            fk_constraints = {}
            for col_name, col_type in table['columns'].items():
                col_type = type_str_to_sql_type[col_type]
                is_primary_key = col_name in table['primary_key']
                if 'foreign_keys' in tables[table_name]:
                    is_foreign_key = col_name in table['foreign_keys']
                    if is_foreign_key:
                        fk_ref_table_name = table['foreign_keys'][col_name].split('.')[0]
                        if fk_ref_table_name not in fk_constraints:
                            fk_constraints[fk_ref_table_name] = {'cols': [], 'cols_refs': []}
                        fk_constraints[fk_ref_table_name]['cols'].append(col_name)
                        fk_constraints[fk_ref_table_name]['cols_refs'].append(table['foreign_keys'][col_name])
                columns.append(
                    Column(
                        col_name,
                        col_type,
                        primary_key=is_primary_key,
                        autoincrement=False,
                    ))
            multi_table_fk_constraints = []
            for tbl, fk_cons in fk_constraints.items():
                multi_table_fk_constraints.append(ForeignKeyConstraint(fk_cons['cols'], fk_cons['cols_refs']))

            table = Table(table_name, self._metadata, *columns, *multi_table_fk_constraints)

        # TODO: modify this part, maybe move to yaml file
        metadata_table_create_query = '''CREATE TABLE __config_blob_tables ( table_name TEXT, blob_key TEXT );'''
        query_list.append(metadata_table_create_query)
        for blob_table in self._config['blob_tables']:
            for blob_key in self._config['blob_tables'][blob_table]:
                insert_query = f'''INSERT INTO __config_blob_tables (table_name, blob_key) VALUES ('{blob_table}', '{blob_key}');'''
                query_list.append(insert_query)
        return query_list


    async def _create_table(self):
        query_list = self.create_table_sql()
        drop = MetaData()

        async with self._sql_engine.begin() as conn:
            await conn.run_sync(drop.reflect)
            await conn.run_sync(drop.drop_all)
            for query in query_list:
                await conn.execute(text(query))
            await conn.run_sync(self._metadata.create_all)

    # TODO: Should we support to insert table without some columns?
    async def insert_table(self, df, table_name):
        def read_cols(conn):
            metadata = MetaData()
            sqlalchemy_table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=conn)
            col_list = []
            for col in sqlalchemy_table.columns:
                col_list.append(col.name)
            return col_list, sqlalchemy_table

        async with self._sql_engine.begin() as conn:
            cols, table = await conn.run_sync(read_cols)
            df = df[cols]
            records = df.to_dict(orient='records')
            await conn.execute(insert(table), records)
