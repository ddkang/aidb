import importlib

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine


class AIDB:
  @staticmethod
  def from_config(config_path):
    config = importlib.import_module(config_path)

    if config.USE_TASTI:
      tasti_config = importlib.import_module()
      aidb_engine = Engine(f'{config.DB_URL}/{config.DB_NAME}', debug=False)
    else:
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
