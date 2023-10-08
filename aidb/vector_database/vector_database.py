import abc
import numpy as np
import pandas as pd


class VectorDatabase(abc.ABC):

  @abc.abstractmethod
  def create_index(
      self,
      index_name: str,
      similarity: str,
      recreate_index: bool = False
  ):
    '''
    Create a new index of vector database
    :param index_name: index name, similar concept to table name in relational database
    :param similarity: similarity function
    :param recreate_index: whether to recreate index
    '''
    raise NotImplementedError


  @abc.abstractmethod
  def delete_index(self, index_name: str):
    '''
    Delete index by index name from vector database if it exists
    '''
    raise NotImplementedError


  @abc.abstractmethod
  def insert_data(self, index_name: str, data: pd.DataFrame):
    '''
    Insert data into index
    :param data: data to be inserted, usually contains id and embedding, metadata is optional
    '''
    raise NotImplementedError


  @abc.abstractmethod
  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray, reload = False) -> np.ndarray:
    '''
    Get embeddings by id and return results
    :param ids: ids of data
    '''
    raise NotImplementedError


  @abc.abstractmethod
  def query_by_embedding(
      self,
      index_name: str,
      query_embeddings: np.ndarray,
      top_k: int = 5
  ) -> (np.ndarray, np.ndarray):
    '''
    Query nearest k embeddings, return topk ids and distances
    :param query_embeddings: embeddings to be queried
    :param top_k: top k nearest neighbors
    '''
    raise NotImplementedError


  @abc.abstractmethod
  def execute(
      self,
      index_name: str,
      embeddings: np.ndarray,
      reps: np.ndarray,
      top_k: int = 5
  ) -> (np.ndarray, np.ndarray):
    '''
    create a new index, query topk representatives and distances for each blob id
    :param embeddings: embeddings for all data
    :param reps: cluster representatives sequential idx
    :param top_k: top k nearest neighbors
    '''
    raise NotImplementedError