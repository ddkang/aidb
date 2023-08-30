import asyncio
from typing import Dict

import sqlalchemy
import sqlalchemy.ext.asyncio

from aidb.config.config import Config
from aidb.config.config_types import (Column, ColumnType, Table,
                                      _get_normalized_column_name)
from aidb.utils.constants import CACHE_PREFIX, CONFIG_PREFIX
from aidb.utils.logger import logger


class BaseEngine():
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      verbose: bool = False,
  ):
    self._connection_uri = connection_uri
    self._verbose = verbose

    self._loop = asyncio.new_event_loop()
    self._dialect = self._infer_dialect(connection_uri)
    self._sql_engine = self._create_sql_engine()

    if infer_config:
      self._config = self._loop.run_until_complete(self._infer_config())


  def __del__(self):
    self._loop.run_until_complete(self._sql_engine.dispose())
    self._loop.close()


  def _infer_dialect(self, connection_uri: str):
    # Conection URIs have the following format:
    # dialect+driver://username:password@host:port/database
    # See https://docs.sqlalchemy.org/en/20/core/engines.html
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
        'echo': self._verbose,
        'max_overflow': -1,
      }
    else:
      kwargs = {}

    engine = sqlalchemy.ext.asyncio.create_async_engine(
      self._connection_uri,
      **kwargs,
    )

    return engine
  

  async def _infer_config(self):
    '''
    Infer the database configuration from the sql engine.
    Extracts:
    - Tables, columns (+ types), and foriegn keys.
    - Cache tables
    - Blob tables
    - Generated columns
    '''
    # We use an async engine, so we need a function that takes in a synchrnous connection
    def config_from_conn(conn):
      inspector = sqlalchemy.inspect(conn)
      table_names = inspector.get_table_names()
      aidb_tables: Dict[str, Table] = {}
      aidb_cols = {}
      aidb_relations = {}

      for table_name in table_names:
        if table_name.startswith(CONFIG_PREFIX):
          continue
        if table_name.startswith(CACHE_PREFIX):
          continue
        columns = inspector.get_columns(table_name)
        pkeys = inspector.get_pk_constraint(table_name)['constrained_columns']
        table_cols = {}
        for column in columns:
          col = Column(
            table_name,
            column['name'],
            ColumnType.parse(column['type']),
            column['name'] in pkeys,
          )
          aidb_cols[col.full_name] = col
          table_cols[col.name] = col

        # Foreign keys
        fkeys = {}
        for fkey in inspector.get_foreign_keys(table_name):
          left_key = _get_normalized_column_name(table_name, fkey['constrained_columns'][0])
          right_key = _get_normalized_column_name(fkey['referred_table'], fkey['referred_columns'][0])
          fkeys[fkey['constrained_columns'][0]] = right_key
          aidb_relations[left_key] = right_key

        aidb_tables[table_name] = Table(
          table_name,
          table_cols,
          pkeys,
          fkeys,
        )

      config = Config(
        self._connection_uri,
        [],
        [],
        aidb_tables,
        aidb_cols,
        aidb_relations,
        {},
        {},
        {},
        {},
      )
      return config


    async with self._sql_engine.begin() as conn:
      config = await conn.run_sync(config_from_conn)

      print(config)

    return config