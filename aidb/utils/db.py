import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap

from aidb.utils.logger import logger


def infer_dialect(connection_uri: str):
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
    logger.warning(
      f'Unsupported dialect: {dialect}. Defaulting to mysql')
    dialect = 'mysql'

  return dialect


def create_sql_engine(connection_uri, debug=False):
  dialect = infer_dialect(connection_uri)
  logger.info(f'Creating SQL engine for {dialect}')
  if dialect == 'mysql':
    kwargs = {
      'echo': debug,
      'max_overflow': -1,
    }
  else:
    kwargs = {}

  engine = sqlalchemy.ext.asyncio.create_async_engine(
    connection_uri,
    **kwargs,
  )

  return engine
