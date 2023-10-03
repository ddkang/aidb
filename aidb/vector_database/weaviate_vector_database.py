from typing import Dict,  Optional
import pandas as pd
import weaviate
from aidb.utils.logger import logger
import numpy as np
from weaviate.util import generate_uuid5
from weaviate import AuthApiKey, AuthClientPassword
from aidb.vector_database.vector_database import VectorDatabase
from aidb.config.config_types import WeaviateAuth

class WeaviateVectorDatabase(VectorDatabase):
  def __init__(self, weaviate_auth: WeaviateAuth):
    '''
    Authentication
    '''
    auth_client_secret = self._get_auth_secret(weaviate_auth.username, weaviate_auth.pwd, weaviate_auth.api_key)
    self.weaviate_client = weaviate.Client(
      url=weaviate_auth.url,
      auth_client_secret=auth_client_secret
    )

    status = self.weaviate_client.is_ready()
    if not status:
      raise Exception('Initial connection to Weaviate failed')

    self.weaviate_client.batch.configure(batch_size=200, dynamic=True)

    self.index_list = [c['class'] for c in self.weaviate_client.schema.get()['classes']]


  @staticmethod
  def _get_auth_secret(username: Optional[str] = None, password: Optional[str] = None, api_key: Optional[str] = None):
    '''
    verify the user information
    '''
    if api_key:
      return AuthApiKey(api_key=api_key)
    elif username and password:
      return AuthClientPassword(username, password)
    else:
      raise Exception('Please provide api or username and password to connect Weaviate')


  def create_index(
    self,
    index_name: str,
    similarity: str = 'l2-squared',
    recreate_index: bool = False
  ):
    '''
    Create a new index of vectordatabase
    :similarity: similarity function, it should be one of l2-squared, cosine and dot
    '''
    if recreate_index:
      self.delete_index(index_name)

    if not index_name:
      raise Exception('Index name must not be none')

    if similarity not in ['cosine', 'dot', 'l2-squared']:
      raise Exception('Similarity function must be one of euclidean, cosine and dotproduct')

    if index_name in self.index_list:
      raise Exception(f'Index {index_name} already exists, please use another name')

    schema = {
      'class': index_name,
      'description': f'Index {index_name} to store embedding',
      'vectorizer': 'none',
      'properties': [{'name': 'original_id', 'dataType': ['int']}],
      'vectorIndexConfig': {'distance': similarity}
    }

    self.weaviate_client.schema.create_class(schema)
    self.index_list.append(index_name)


  def delete_index(self, index_name: str):
    '''
    delete an index
    '''
    if index_name in self.index_list:
      self.weaviate_client.schema.delete_class(index_name)
      self.index_list.remove(index_name)
      logger.info("Index '%s' deleted.", index_name)


  def _sanitize_index_name(self, index_name: str) -> str:
    '''
    index should start with a capital
    '''
    return index_name[0].upper() + index_name[1:]


  def _check_index_validity(self, index_name: str):

    index_name = self._sanitize_index_name(index_name)
    if index_name not in self.index_list:
      raise Exception(f'Couldn\'t find index {index_name}, please create it first')
    return index_name


  def insert_data(self, index_name: str, data: pd.DataFrame):
    '''
    insert data into an index
    '''
    index_name = self._check_index_validity(index_name)

    with self.weaviate_client.batch as batch:
      for _, item in data.iterrows():
        metadata = dict()
        metadata['original_id'] = item['id']
        uuid = generate_uuid5(item['id'])
        vector = item['values']
        batch.add_data_object(metadata, index_name, uuid, vector)


  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray, reload = False) -> np.ndarray:
    '''
    Get data by id and return results
    '''
    index_name = self._check_index_validity(index_name)
    result = []
    for id in ids:
      uuid = generate_uuid5(id)
      fetch_response = self.weaviate_client.data_object.get(uuid=uuid, with_vector=True, class_name=index_name)
      result.append(fetch_response['vector'])
    return np.array(result)


  def query_by_embedding(
      self,
      index_name: str,
      query_emb_list: np.ndarray,
      top_k: int = 5,
      filters: Optional[Dict[str, str]] = None
  ) -> (np.ndarray, np.ndarray):
    '''
    Query nearest k embeddings, return embeddings and ids
    :param filters: do filter by metadata
    '''
    index_name = self._check_index_validity(index_name)
    properties = ['original_id', '_additional {distance}']
    multi_query = []
    alias_list = ['alias' + str(i) for i in range(len(query_emb_list))]
    for alias, query_emb in zip(alias_list, query_emb_list):
      new_query = self.weaviate_client.query.get(class_name=index_name, properties=properties)
      new_query.with_near_vector({'vector': query_emb})
      new_query.with_limit(top_k)
      new_query.with_alias(alias)
      if filters:
        new_query.with_where(filters)
      multi_query.append(new_query)

    response = self.weaviate_client.query.multi_get(multi_query).do()
    response = response['data']['Get']

    all_topk_reps, all_topk_dists = [], []
    for alias in alias_list:
      topk_rep, topk_dist = [], []
      for record in response[alias]:
        topk_rep.append(record['original_id'])
        topk_dist.append(record['_additional']['distance'])
      all_topk_reps.append(topk_rep)
      all_topk_dists.append(topk_dist)

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
