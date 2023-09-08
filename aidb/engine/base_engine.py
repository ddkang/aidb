import asyncio
from typing import List, Tuple

import pandas as pd
import sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.ext.automap

from aidb.config.config import Config
from aidb.config.config_types import Graph, InferenceBinding
from aidb.inference.bound_inference_service import BoundInferenceService, CachedBoundInferenceService
from aidb.inference.inference_service import InferenceService
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


  # ---------------------
  # Setup
  # ---------------------
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
      config = Config(
        {},
        [],
        self._connection_uri,
        None,
        None,
        None,
        None,
        None,
      )
      config.load_from_sqlalchemy(conn)
      return config


    async with self._sql_engine.begin() as conn:
      config: Config = await conn.run_sync(config_from_conn)

    if self._debug:
      import prettyprinter as pp
      pp.install_extras(exclude=['django', 'ipython', 'ipython_repr_pretty'])
      pp.pprint(config)
      print(config.blob_tables)

    return config


  def register_inference_service(self, service: InferenceService):
    self._config.add_inference_service(service.name, service)


  def bind_inference_service(self, service_name: str, binding: InferenceBinding):
    bound_service = CachedBoundInferenceService(
      self._config.inference_services[service_name],
      binding,
      self._loop,
      self._sql_engine,
      self._config.columns,
      self._config.tables,
      self._dialect,
    )
    self._config.bind_inference_service(bound_service)


  # ---------------------
  # Properties
  # ---------------------
  @property
  def dialect(self):
    return self._dialect


  # ---------------------
  # Inference
  # ---------------------
  def prepare_multitable_inputs(self, raw_inputs: List[Tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    '''
    Prepare the inputs for inference.
    '''
    assert len(raw_inputs) >= 1
    final_df = raw_inputs[0][1]
    for idx, (table_name, df) in enumerate(raw_inputs[1:]):
      last_table_name = raw_inputs[idx][0]
      table_relations = self._config.relations_by_table[table_name]
      join_keys = [fk for fk in table_relations if fk.startswith(last_table_name)]
      final_df = final_df.merge(df, on=join_keys, how='inner')

    return final_df


  def process_inference_outputs(self, binding: InferenceBinding, joined_outputs: pd.DataFrame) -> pd.DataFrame:
    '''
    Process the outputs of inference by renaming the columns and selecting the
    output columns.
    '''
    df_cols = list(joined_outputs.columns)
    for idx, col in enumerate(binding.output_columns):
      joined_outputs.rename(columns={df_cols[idx]: col}, inplace=True)
    res = joined_outputs[list(binding.output_columns)]
    return res


  def inference(self, inputs: pd.DataFrame, bound_service: BoundInferenceService) -> List[pd.DataFrame]:
    return bound_service.batch(inputs)


  def execute(self, query: str):
    raise NotImplementedError()