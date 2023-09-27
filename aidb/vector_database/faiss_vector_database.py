from typing import Optional
from aidb.vector_database.vector_database import VectorDatabase
import pandas as pd
import faiss
from aidb.utils.logger import logger
import numpy as np


class FaissVectorDataBase(VectorDatabase):
  def __init__(self, path: str):
    '''
    Authentication
    '''

    self.index_list = dict()
    self.path = path


  def create_index(
    self,
    index_name: str,
    embedding_dim: int,
    similarity: str = 'l2',
    index_factory: str = 'Flat',
    n_links: int = 64,
    ef_search: int = 20,
    ef_construction: int = 80,
    recreate_index: bool = False
  ):
    '''
    Create a new index, which is similar to a table
    '''
    if recreate_index:
      self.delete_index(index_name)

    if not index_name:
      raise Exception(f'Index name must not be none')

    if similarity in ("dot_product", "cosine"):
      metric_type = faiss.METRIC_INNER_PRODUCT
    elif similarity == "l2":
      metric_type = faiss.METRIC_L2
    else:
      raise Exception('Similarity function must be one of l2, cosine and dot_product')

    if index_factory == "HNSW":
      new_index = faiss.IndexHNSWFlat(embedding_dim, n_links, metric_type)
      new_index.hnsw.efSearch = ef_search
      new_index.hnsw.efConstruction = ef_construction
    else:
      new_index = faiss.index_factory(embedding_dim, index_factory, metric_type)

    # use to add data with ids
    self.index_list[index_name] = new_index


  def load_index(self, index_name: str):
    '''
    Read index from disk
    '''
    self.index_list[index_name] = faiss.read_index(self.path)


  def save_index(self, index_name:str):
    faiss.write_index(self.index_list[index_name], self.path)


  def delete_index(self, index_name: str):
    '''
    delete an index
    '''
    if index_name in self.index_list:
      del self.index_list[index_name]
      logger.info("Index '%s' deleted.", index_name)


  def _connect_by_index(self, index_name: str):

    if index_name not in self.index_list:
      raise Exception(f'Couldn\'t find index {index_name}, please create it first')

    return self.index_list[index_name]


  def insert_data(self, index_name: str, data: pd.DataFrame):
    '''
    insert data into an index
    '''
    connected_index = self._connect_by_index(index_name)

    if not connected_index.is_trained:
      raise Exception(f'FAISS index of type {connected_index} must be trained before adding vectors')

    embedding_list = np.array(list(data['values'])).astype('float32')

    self.index_list[index_name].add(embedding_list)


  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray) -> np.ndarray:
    '''
    Get data by id and return results
    '''
    connected_index = self._connect_by_index(index_name)

    result = []
    for id in ids.tolist():
      record = connected_index.reconstruct(id)
      result.append(record)
    return np.array(result)


  def query_by_embedding(
    self,
    index_name: str,
    query_embeddings: np.ndarray,
    top_k: int = 5,
    filter_ids: Optional[np.ndarray] = None
  ) -> (np.ndarray, np.ndarray):
    '''
    Query nearest k embeddings, return embeddings and ids
    '''
    connected_index = self._connect_by_index(index_name)

    params = None
    if filter_ids is not None:
      id_selector = faiss.IDSelectorArray(filter_ids)
      params = faiss.SearchParametersIVF(sel=id_selector)
    all_topk_dists, all_topk_reps = connected_index.search(query_embeddings.astype('float32'), top_k, params=params)

    return np.array(all_topk_reps).astype('int64'), np.array(all_topk_dists).astype('float32')


  def train_index(
    self,
    index_name: str,
    embeddings: Optional[np.ndarray] = None
  ):

    connected_index = self._connect_by_index(index_name)
    connected_index.train(embeddings)


  def execute(self,
              index_name: str,
              embeddings: np.ndarray,
              reps: np.ndarray,
              top_k: int = 5
              ) -> (np.ndarray, np.ndarray):
    '''
    create index for cluster representatives, get topk representatives and distances for each blob index
    '''
    #TODO: maybe also change to create an individual index
    self.create_index(index_name, embeddings.shape[1])
    data = pd.DataFrame({'values': embeddings.tolist()})
    self.insert_data(index_name, data)
    topk_reps, topk_dists = self.query_by_embedding(index_name, embeddings, top_k=top_k, filter_ids=reps)
    self.save_index(index_name)
    return topk_reps, topk_dists
