import importlib
import pandas as pd

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine
# from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDatabase
from aidb.vector_database.tasti import Tasti


def get_tasti_config(tasti_config):
  vector_database = {
    'FAISS': FaissVectorDatabase,
    # 'CHROMA': ChromaVectorDatabase,
    'WEAVIATE': WeaviateVectorDatabase
  }
  vector_database_config = tasti_config.vector_database
  vector_database_type = vector_database_config['vector_database_type'].upper()
  try:
    user_vector_database = vector_database[vector_database_type](**vector_database_config['auth'])
  except KeyError:
    raise ValueError(f'{vector_database_type} is not a supported type. We support FAISS, Chroma and Weaviate.')
  tasti_index = Tasti(vector_database=user_vector_database, **tasti_config.tasti_engine)

  selected_vector_id_df = None

  if hasattr(tasti_config, 'vector_id_csv') and tasti_config.vector_id_csv is not None:
    selected_vector_id_df = pd.read_csv(tasti_config.vector_id_csv)
    if len(selected_vector_id_df.columns) != 1:
      raise Exception('Vector id csv file should contain one column for vector id')
    selected_vector_id_df.columns.values[0] = 'vector_id'

  return tasti_index, selected_vector_id_df


class AIDB:
  @staticmethod
  def from_config(config_path, verbose=False):
    config = importlib.import_module(config_path)

    if hasattr(config, 'INITIALIZE_TASTI') and config.INITIALIZE_TASTI is True:
      tasti_index, vector_id_df = get_tasti_config(config)
      aidb_engine = Engine(
          f'{config.DB_URL}/{config.DB_NAME}',
          debug=False,
          tasti_index=tasti_index,
          user_specified_vector_ids=vector_id_df
      )
    else:
      aidb_engine = Engine(f'{config.DB_URL}/{config.DB_NAME}', debug=False)

    for inference_engine in config.inference_engines:
      service = inference_engine["service"]
      input_col = inference_engine["input_col"]
      output_col = inference_engine["output_col"]
      aidb_engine.register_inference_service(service)
      aidb_engine.bind_inference_service(
        service.name,
        InferenceBinding(input_col, output_col),
        verbose)
    return aidb_engine
