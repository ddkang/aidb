import chromadb
import numpy as np
import pandas as pd

from typing import Dict, Optional

from aidb.utils.logger import logger
from aidb.vector_database.vector_database import VectorDatabase


class ChromaVectorDatabase(VectorDatabase):
  def __init__(self, path: str):
    '''
    Authentication
    :param path: path to store vector database
    '''
    self.path = path
    self.client = chromadb.PersistentClient(path=path)
    self.index_list = [collection.name for collection in self.client.list_collections()]


  def create_index(
      self,
      index_name: str,
      similarity: str = 'l2',
      recreate_index: bool = False
  ):
    '''
    Create a new index of vector database
    :similarity: similarity function, it should be one of l2, cosine and ip
    '''
    if recreate_index:
      self.delete_index(index_name)

    if not index_name:
      raise Exception(f'Index name must not be none')

    if similarity not in ['l2', 'cosine', 'ip']:
      raise Exception('Similarity function must be one of euclidean, cosine and dotproduct')

    metadata = {'hnsw:space': similarity}

    if index_name in self.index_list:
      raise Exception(f'Index {index_name} already exists, please use another name')
    else:
      self.client.create_collection(name=index_name, metadata=metadata)
      self.index_list.append(index_name)


  def delete_index(self, index_name: str):
    '''
    delete an index
    '''
    if index_name in self.index_list:
      self.client.delete_collection(index_name)
      self.index_list.remove(index_name)
      logger.info("Index '%s' deleted.", index_name)


  def _connect_by_index(self, index_name: str):

    if index_name not in self.index_list:
      raise Exception(f'Couldn\'t find index {index_name}, please create it first')

    return self.client.get_collection(index_name)


  def insert_data(self, index_name: str, data: pd.DataFrame):
    '''
    insert data into an index
    '''
    connected_index = self._connect_by_index(index_name)

    ids = np.array(data['id']).astype('str').tolist()
    embeddings = data['values'].tolist()
    metadata = data.drop(['id', 'values'], axis=1)
    if metadata.empty:
      metadata = None
    connected_index.upsert(ids=ids, embeddings=embeddings, metadatas=metadata)


  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray, reload = False) -> np.ndarray:
    '''
    Get data by id and return results
    '''
    if reload:
      self.client = chromadb.PersistentClient(path=self.path)
    connected_index = self._connect_by_index(index_name)
    id_list = ids.astype('str').tolist()
    fetch_response = connected_index.get(ids=id_list, include=['embeddings'])
    result = np.array(fetch_response['embeddings'])
    return result


  def query_by_embedding(
      self,
      index_name: str,
      query_embeddings: np.ndarray,
      top_k: int = 5,
      filters: Optional[Dict[str, str]] = None
  ) -> (np.ndarray, np.ndarray):
    '''
    Query nearest k embeddings, return embeddings and ids
    '''
    connected_index = self._connect_by_index(index_name)
    response = connected_index.query(query_embeddings=query_embeddings.tolist(), n_results=top_k, where=filters)
    all_topk_reps, all_topk_dists = response['ids'], response['distances']

    return np.array(all_topk_reps).astype('int64'), np.array(all_topk_dists)


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