import importlib
import asyncio

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine
from tests.utils import setup_aidb_engine


async def one_time_aidb_setup(DB_URL, DB_NAME, DATA_DIR):
  await setup_aidb_engine(DB_URL, DB_NAME, DATA_DIR)


class AIDB:
  @staticmethod
  def from_config(config_path):
    config = importlib.import_module(config_path)

    asyncio.run(one_time_aidb_setup(config.DB_URL, config.DB_NAME, config.DATA_DIR))

    aidb_engine = Engine(f'{config.DB_URL}/{config.DB_NAME}', debug=False)

    inference_engine = config.inference_engine
    aidb_engine.register_inference_service(inference_engine)
    aidb_engine.bind_inference_service(inference_engine.name,
                                       InferenceBinding(config.input_col, config.output_col))
    return aidb_engine
