import numpy as np
import abc
import pandas as pd


class VectorDatabase(abc.ABC):
  index_name: str

  @abc.abstractmethod
  def create_index(
    self,
    index_name: str,
    similarity: str,
    recreate_index: bool = False
  ):
    pass


  @abc.abstractmethod
  def delete_index(self, index_name: str):
    pass


  @abc.abstractmethod
  def insert_data(self, index_name: str, data: pd.DataFrame):
    pass


  @abc.abstractmethod
  def get_embeddings_by_id(self, index_name: str, ids: np.ndarray) -> np.ndarray:
    pass


  @abc.abstractmethod
  def query_by_embedding(self,
                         index_name: str,
                         query_embeddings: np.ndarray,
                         top_k: int = 5
                         ) -> (np.ndarray, np.ndarray):
    pass


  @abc.abstractmethod
  def execute(self,
              index_name: str,
              embeddings: np.ndarray,
              reps: np.ndarray,
              top_k: int = 5
              ) -> (np.ndarray, np.ndarray):
    pass