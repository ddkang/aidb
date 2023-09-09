import asyncio

import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap

from sqlalchemy.sql import text

from aidb.utils.logger import logger
from collections import defaultdict
import pandas as pd

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
        query_list = []
        tables = self._config['tables']
        for table_name in tables:
            sql_list = []
            col_list = []
            for col_name, col_type in tables[table_name]['columns'].items():
                if col_type == 'str':
                    col_type = 'TEXT'
                col_type = col_type.upper()
                col_list.append(f'{col_name} {col_type}')
            sql_list.append(', '.join(col_list))

            pks = ', '.join(tables[table_name]['primary_key'])
            sql_list.append(f'PRIMARY KEY ({pks})')

            if 'foreign_keys' in tables[table_name]:
                fk_list = []
                refer_table_dict = defaultdict(list)
                for fk, refer_table_col in tables[table_name]['foreign_keys'].items():
                    refer_table = refer_table_col.split('.')[0]
                    refer_col = refer_table_col.split('.')[1]
                    refer_table_dict[refer_table].append((fk, refer_col))

                for refer_table in refer_table_dict:
                    fks = ', '.join([x[0] for x in refer_table_dict[refer_table]])
                    refer_cols = ', '.join([x[1] for x in refer_table_dict[refer_table]])
                    fk_list.append(f'FOREIGN KEY ({fks}) REFERENCES {refer_table}({refer_cols})')
                sql_list.append(', '.join(fk_list))

            sql = ', '.join(sql_list)

            drop_query = f'''DROP TABLE IF EXISTS {table_name};'''
            create_query = f'''CREATE TABLE {table_name} ({sql});'''
            query_list.insert(0, drop_query)
            query_list.append(create_query)

        # Create metadata table
        metadata_table_drop_query = '''DROP TABLE IF EXISTS aidb.__config_blob_tables;'''
        metadata_table_create_query = '''CREATE TABLE aidb.__config_blob_tables ( table_name TEXT, blob_key TEXT );'''
        query_list.append(metadata_table_drop_query)
        query_list.append(metadata_table_create_query)
        for blob_table in self._config['blob_tables']:
            for blob_key in self._config['blob_tables'][blob_table]:
                insert_query = f'''INSERT INTO aidb.__config_blob_tables (table_name, blob_key) VALUES ('{blob_table}', '{blob_key}');'''
                query_list.append(insert_query)

        return query_list

    async def _create_table(self):
        query_list = self.create_table_sql()
        async with self._sql_engine.begin() as conn:
            for query in query_list:
                await conn.execute(text(query))

    # TODO: Should we support to insert table without some columns?
    async def insert_value(self, df, table_name):
        def read_cols(conn):
            metadata = sqlalchemy.MetaData()
            sqlalchemy_table = sqlalchemy.Table(table_name, metadata , autoload=True , autoload_with=conn)
            col_list = []
            for col in sqlalchemy_table.columns:
                col_list.append(col.name)
            return col_list

        async with self._sql_engine.begin() as conn:
            cols = await conn.run_sync(read_cols)
            values = [':' + col for col in cols]
            col_sql = ', '.join(cols)
            value_sql = ', '.join(values)
            sql = f'''INSERT INTO {table_name} ({col_sql}) VALUES ({value_sql})'''
            for index, row in df.iterrows():
                params = dict()
                for col in cols:
                    params[col] = row[col]
                await conn.execute(text(sql), params)
