import pandas as pd
from numba import njit, prange
from tqdm import tqdm
from aidb.vector_database.faiss_vector_database import FaissVectorDataBase
from aidb.vector_database.chroma_vector_database import ChromaVectorDataBase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDataBase
from typing import List, Optional
import numpy as np

@njit(parallel=True)
def get_and_update_dists(x: List[float], embeddings: np.ndarray, min_dists: np.ndarray):
  for i in prange(len(embeddings)):
    dists = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    if dists < min_dists[i]:
      min_dists[i] = dists


class Tasti:
  def __init__(self,
               index_name: str,
               blob_index: pd.DataFrame,
               nb_buckets: int,
               vector_database: str = 'FAISS',
               url: Optional[str] = None,
               username: Optional[str] = None, #FIXME: define a datatype for weaviate input
               pwd: Optional[str] = None,
               api_key: Optional[str] = None,
               index_path: Optional[str] = None,
               percent_fpf: float = 0.75,
               seed: int = 1234):

    self.index_name = index_name
    self.rep_index_name = self.index_name + '__representatives'
    self.blob_index = blob_index
    self.nb_buckets = nb_buckets
    self.percent_fpf = percent_fpf
    self.seed = seed
    # FAISS query embedding distances by doing filter, other vector database create individual index
    self.do_filter = False

    if vector_database == 'FAISS':
      if index_path is None:
        raise Exception('FAISS requires index path')
      self.vector_database = FaissVectorDataBase(index_path)
      self.vector_database.load_index(index_name)
      self.do_filter = True
    elif vector_database == 'Chroma':
      if index_path is None:
        raise Exception('Chroma requires index path')
      self.vector_database = ChromaVectorDataBase(index_path)
    elif vector_database == 'Weaviate':
      if url is None:
        raise Exception('Weaviate requires url to connect')
      self.vector_database = WeaviateVectorDataBase(url, username, pwd, api_key)
    else:
      raise Exception(f'{vector_database} is not supported, please use FAISS, Chroma or Weaviate')

    if self.index_name not in self.vector_database.index_list:
      raise Exception(f'Index {index_name} doesn\'t exist in vector database')

    self.embeddings = self.vector_database.get_embeddings_by_id(self.index_name,
                                                                self.blob_index.values.reshape(1, -1)[0])
    self.reps = None


  def _FPF(self, nb_buckets: Optional[int] = None) -> np.ndarray:
    '''
    FPF mining algorithm and return cluster representative ids, old cluster representatives will always be kept
    '''
    if nb_buckets is not None:
      buckets = nb_buckets
    else:
      buckets = self.nb_buckets

    np.random.RandomState(self.seed)
    reps = np.full(buckets, -1)
    min_dists = np.full(len(self.embeddings), np.Inf, dtype=np.float32)
    num_random = int((1 - self.percent_fpf) * len(reps))
    random_reps = np.random.choice(len(self.embeddings), num_random, replace=False)

    reps[0] = random_reps[0]
    get_and_update_dists(self.embeddings[reps[0]], self.embeddings, min_dists)

    for i in tqdm(range(1, num_random), desc='RandomBucketter'):
      reps[i] = random_reps[i]
      get_and_update_dists(self.embeddings[reps[i]], self.embeddings, min_dists)

    for i in tqdm(range(num_random, buckets), desc='FPFBucketter'):
      reps[i] = np.argmax(min_dists)
      get_and_update_dists(self.embeddings[reps[i]], self.embeddings, min_dists)

    self.reps = np.unique(np.concatenate((self.reps, reps)))


  def get_representative_blob_ids(self) -> pd.DataFrame:

    if self.reps is None:
      self._FPF()
    return self.blob_index.iloc[self.reps]


  def get_topk_representatives_for_all(self, top_k: int = 5):
    '''
    get topk representatives and distances for all blob index
    '''
    if self.reps is None:
      self._FPF()

    topk_reps, topk_dists = self.vector_database.execute(self.rep_index_name, self.embeddings, self.reps, top_k)
    topk_reps = self.blob_index.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=self.blob_index.squeeze())


  def get_topk_representatives_for_new_embeddings(self,
                                                  new_blob_index: pd.DataFrame,
                                                  top_k: int = 5
                                                  ):
    '''
    get topk representatives and distances for new embeddings using stale cluster representatives,
    in other words, we don't need to use FPF to reselect cluster representatives
    '''
    new_embeddings = self.vector_database.get_embeddings_by_id(self.index_name,
                                                               new_blob_index.values.reshape(1, -1)[0])
    if self.do_filter:
      topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name,
                                                                      new_embeddings,
                                                                      top_k,
                                                                      filters=self.reps)
    else:
      topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name, new_embeddings, top_k)
    topk_reps = self.blob_index.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=new_blob_index.squeeze())


  def update_topk_representatives_for_all(self,
                                          new_blob_index: pd.DataFrame,
                                          top_k: int = 5,
                                          nb_buckets: Optional[int] = None
                                          ):
    '''
    when new embeddings are added, we update cluster representative ids and dists for all blob index
    '''
    #TODO: do we need to check if there is override between blob_index and new_blob_index?
    self.blob_index = pd.concat([self.blob_index, new_blob_index])
    new_embeddings = self.vector_database.get_embeddings_by_id(self.index_name,
                                                               new_blob_index.values.reshape(1, -1)[0])
    self.embeddings = np.concatenate((self.embeddings, new_embeddings), axis=0)

    self._FPF(nb_buckets)
    return self.get_topk_representatives_for_all(top_k)