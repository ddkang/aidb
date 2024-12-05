import pandas as pd

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine
from aidb.inference.examples.llm_inference_service import LLMInference
from aidb.inference.examples.vllm_inference_service import VLLMInference
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDatabase
from aidb.vector_database.tasti import Tasti

def get_tasti_config(tasti_config):
  vector_database = {
    'FAISS': FaissVectorDatabase,
    'CHROMA': ChromaVectorDatabase,
    'WEAVIATE': WeaviateVectorDatabase
  }

  vector_database_type = tasti_config['type'].upper()
  try:
    user_vector_database = vector_database[vector_database_type](**tasti_config['auth'])
  except KeyError:
    raise ValueError(f'{vector_database_type} is not a supported type. We support FAISS, Chroma and Weaviate.')
  tasti_index = Tasti(vector_database=user_vector_database, **tasti_config['tasti_engine'])

  selected_vector_id_df = None

  if 'vector_id_csv' in tasti_config:
    selected_vector_id_df = pd.read_csv(tasti_config['vector_id_csv'])
    if len(selected_vector_id_df.columns) != 1:
      raise Exception('Vector id csv file should contain one column for vector id')
    selected_vector_id_df.columns.values[0] = 'vector_id'

  return tasti_index, selected_vector_id_df


def setup_inference(service_name, service, service_config):
  inference_dict ={
    'LLM': LLMInference,
    'VLLM': VLLMInference
  }
  service_config['name'] = service_name
  return inference_dict[service.upper()](**service_config)
  
  
class AIDB:
  @staticmethod
  def from_config(config, verbose=False):
    db_config = f"{config['db_config']['url']}/{config['db_config']['name']}"
    if 'vector_database' in config: 
      tasti_index, vector_id_df = get_tasti_config(config['vector_database'])
      aidb_engine = Engine(
          db_config,
          debug=False,
          tasti_index=tasti_index,
          user_specified_vector_ids=vector_id_df
      )
    else:
      aidb_engine = Engine(db_config, debug=False)

    for inference_engine in config['services']:
      service = setup_inference(inference_engine['name'], inference_engine["service"], inference_engine['service_config'])
      copy_map = inference_engine.get("copy", {})
      aidb_engine.register_inference_service(service)
      aidb_engine.bind_inference_service(
        service.name,
        InferenceBinding(tuple(inference_engine["input_cols"]), tuple(inference_engine["output_cols"])),
        copy_map,
        verbose)
    return aidb_engine
