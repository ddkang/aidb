import uuid
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import marqo
from typing import Dict,  Optional

from aidb.utils.logger import logger
from aidb.vector_database.vector_database import VectorDatabase

@dataclass
class MarqoAuth:
  """
  :param url: marqo url
  :param username: marqo username
  :param pwd: marqo password
  :param api_key: marqo api key, user should choose input either username/pwd or api_key
  """
  url: Optional[str] = field(default=None)
  username: Optional[str] = field(default=None)
  pwd: Optional[str] = field(default=None)
  api_key: Optional[str] = field(default=None)

class MarqoVectorDatabase(VectorDatabase):
  def __init__(self, marqo_auth: MarqoAuth, embedding_dim: int) -> None:
    '''
    Authentication
    '''
    if marqo_auth.url is None:
      raise ValueError('Marqo requires URL to connect')
    
    self.marqo_client = marqo.Client(
      url=marqo_auth.url, main_user=marqo_auth.username, main_password=marqo_auth.pwd)
    status = self.marqo_client.get_marqo()
    if not status:
      raise Exception('Initial connection to Marqo failed')
    

  @staticmethod
  def _get_auth_secret(username: Optional[str] = None, password: Optional[str] = None, api_key: Optional[str] = None):
    '''
    verify the user information
    '''
    if api_key:
      return api_key
    elif username and password:
      return (username, password)
    else:
      raise Exception('Please provide api or username and password to connect Marqo')
    
  
  def create_index(
    self,
    embedding_dim: int,
    index_name: str,
    similarity: str = 'l2',
    recreate_index: bool = False
  ):
    '''
    Create a new index
    :similarity: similarity function, it should be one of l1, l2, linf and cosinesiml
    '''
    if recreate_index:
      if index_name in self.index_list:
        self.delete_index(index_name)

    self.index_list = [i.index_name for i in self.marqo_client.get_indexes()['results']]
    self.embedding_dim = embedding_dim

    if not index_name:
      raise Exception('Index name must not be none')

    if similarity not in ['cosinesimil', 'l1', 'l2', 'linf']:
      raise Exception('Similarity function must be one of l1, l2, linf and cosinesiml')

    if index_name in self.index_list:
      raise Exception(f'Index {index_name} already exists, please use another name')

    # Not using any model, providing own vectors
    # https://docs.marqo.ai/1.4.0/API-Reference/Indexes/create_index/#no-model
    self.marqo_client.create_index(
      index_name=index_name,
      settings_dict={
                'index_defaults': {
                    'model': 'no_model',
                    'model_properties': {
                        'dimensions': self.embedding_dim
                    },
                    'ann_parameters':{
                        'space_type': similarity
                    }
                }
            }
      )
    self.index_list.append(index_name)


  def load_index(self):
    '''
    Reload index from Marqo
    '''
    self.index_list = [i.index_name for i in self.marqo_client.get_indexes()['results']]


  def delete_index(self, index_name: str):
    '''
    delete an index
    '''
    if index_name in self.index_list:
      self.marqo_client.delete_index(index_name)
      self.index_list = [i.index_name for i in self.marqo_client.get_indexes()['results']]
      logger.info("Index '%s' deleted.", index_name)
  

  def _check_index_validity(self, index_name: str):
    if index_name not in self.index_list:
      raise Exception(f'Couldn\'t find index {index_name}, please create it first')
    return index_name
  

  def insert_data(self, index_name: str, data: pd.DataFrame):
    '''
    insert data into an index
    '''
    index_name = self._check_index_validity(index_name)

    # Data will have list of dictionaries 
    # with 'id' and 'values' as keys.
    data = data.to_dict(orient="records")
    
    mod_data = []
    for d in data:
      _id = str(d['id'])
      _vec = d['values']
      mod_data.append(
        {
          '_id': _id,
          'aidb_data':{
            'vector': _vec
          }
        }
      )
    response = self.marqo_client.index(
        index_name=index_name
        ).add_documents(
          documents=mod_data,
          mappings={
            'aidb_data':{
              'type': 'custom_vector'
            }
          },
          tensor_fields=['aidb_data'],
          auto_refresh=True,
          client_batch_size=64
        )
    

  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray, reload = False) -> np.ndarray:
    '''
    Get data by id and return results
    '''
    if reload:
      self.load_index()
    index_name = self._check_index_validity(index_name)
    result = []
    response = self.marqo_client.index(index_name).get_documents(document_ids=[str(i) for i in ids], expose_facets=True)
    result = [ii['_tensor_facets'][0]['_embedding'] for ii in response['results']]
    return np.array(result)
  

  def query_by_embedding(
      self,
      index_name: str,
      query_emb_list: np.ndarray,
      top_k: int = 5,
  ) -> (np.ndarray, np.ndarray):
    '''
    Query nearest k embeddings, return embeddings and ids
    '''
    index_name = self._check_index_validity(index_name)
    
    all_topk_reps, all_topk_dists = [], []
    for query_emb in query_emb_list:

      # No query needs to be provided when no_model and custom vectors,
      # only context - https://docs.marqo.ai/1.4.0/API-Reference/Indexes/create_index/#no-model
      response = self.marqo_client.index(index_name).search(
        context={
        'tensor':[{'vector': list(query_emb), 'weight' : 1}],
        }, limit=top_k
      )
      ids = []
      dists = []
      for res in response['hits']:
        ids.append(res['_id'])

        # Because it is similarity score
        dists.append(1-res['_score']) 
      
      all_topk_reps.append(ids)
      all_topk_dists.append(dists)

    return np.array(all_topk_reps), np.array(all_topk_dists)
  

  def execute(
      self,
      index_name: str,
      embeddings: np.ndarray,
      reps: np.ndarray,
      top_k: int = 5
  ) -> (np.ndarray, np.ndarray):
    '''
    create a new index storing cluster representatives, get topk representatives and distances for each blob index
    '''
    self.create_index(index_name, recreate_index=True)
    data = pd.DataFrame({'id': reps.tolist(), 'values': embeddings[reps].tolist()})
    self.insert_data(index_name, data)
    topk_reps, topk_dists = self.query_by_embedding(index_name, embeddings, top_k=top_k)

    return topk_reps, topk_dists

