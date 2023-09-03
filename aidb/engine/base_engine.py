import asyncio
from typing import Dict

import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap

from aidb.config.config import Config
from aidb.config.config_types import Table
from aidb.utils.constants import (BLOB_TABLE_NAMES_TABLE, CACHE_PREFIX,
                                  CONFIG_PREFIX)
from aidb.utils.logger import logger


class BaseEngine():
  def __init__(
      self,
      connection_uri: str,
      infer_config: bool = True,
      debug: bool = False,
  ):
    self._connection_uri = connection_uri
    self._debug = debug

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
  

  async def _infer_config(self) -> Config:
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
      metadata = sqlalchemy.MetaData()
      # TODO: should base go into config?
      Base = sqlalchemy.ext.automap.automap_base()
      Base.prepare(conn, reflect=True)

      aidb_tables: Dict[str, Table] = {}
      aidb_cols = {}
      aidb_relations = {}

      for table_name in Base.classes.keys():
        if table_name.startswith(CONFIG_PREFIX):
          continue
        if table_name.startswith(CACHE_PREFIX):
          continue

        sqlalchemy_table = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=conn)
        table_cols = {}
        for column in sqlalchemy_table.columns:
          aidb_cols[str(column)] = column
          table_cols[column.name] = column

        aidb_tables[table_name] = Table(
          sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=conn),
        )

      blob_metadata_table = sqlalchemy.Table(BLOB_TABLE_NAMES_TABLE, metadata, autoload=True, autoload_with=conn)
      try:
        blob_keys = conn.execute(blob_metadata_table.select()).fetchall()
        blob_tables = list(set([row['table_name'] for row in blob_keys]))
        blob_keys = [f'{row["table_name"]}.{row["blob_key"]}' for row in blob_keys]
      except:
        raise ValueError(f'Could not find blob metadata table {BLOB_TABLE_NAMES_TABLE} or table is malformed')

      # TODO: load metadata columns
      # TODO: check the engines
      # TODO: check that the cache tables are valid

      config = Config(
        self._connection_uri,
        blob_tables,
        blob_keys,
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
      config: Config = await conn.run_sync(config_from_conn)

    if self._debug:
      import prettyprinter as pp
      pp.install_extras(exclude=['django', 'ipython', 'ipython_repr_pretty'])
      pp.pprint(config)
      print(config.blob_tables)

    return config