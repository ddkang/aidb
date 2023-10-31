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

    for inference_engine in config.inference_engines:
      service = inference_engine["service"]
      input_col = inference_engine["input_col"]
      output_col = inference_engine["output_col"]
      aidb_engine.register_inference_service(service)
      aidb_engine.bind_inference_service(
        service.name,
        InferenceBinding(input_col, output_col))
    return aidb_engine