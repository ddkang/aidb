import pandas as pd
from numba import njit, prange
from tqdm import tqdm
from aidb.vector_database.faiss_vector_database import FaissVectorDataBase
from aidb.vector_database.chroma_vector_database import ChromaVectorDataBase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDataBase
from typing import Optional
import numpy as np

@njit(parallel=True)
def get_and_update_dists(x: np.ndarray, embeddings: np.ndarray, min_dists: np.ndarray):
  '''
  :param x: embedding of cluster representatives
  :param embeddings: embeddings of all data
  :min_dists: array to record the minimum distance for each embedding to the embedding of cluster representatives
  '''
  for i in prange(len(embeddings)):
    dists = np.sqrt(np.sum((x - embeddings[i]) ** 2))
    if dists < min_dists[i]:
      min_dists[i] = dists


class Tasti:
  def __init__(
      self,
      index_name: str,
      blob_ids: pd.DataFrame,
      nb_buckets: int,
      vector_database: str = 'FAISS',
      url: Optional[str] = None,
      username: Optional[str] = None, #FIXME: define a datatype for weaviate input
      pwd: Optional[str] = None,
      api_key: Optional[str] = None,
      index_path: Optional[str] = None,
      percent_fpf: float = 0.75,
      seed: int = 1234
  ):
    '''
    :param index_name: vector database index name
    :param blob_ids: blob index in blob table, it should be unique for each data record
    :param nb_buckets: number of buckets for FPF, it should be same as the number of buckets for oracle
    :param vector_database: vector database type, it should be FAISS, Chroma or Weaviate
    :param url: weaviate url
    :param username: weaviate username
    :param pwd: weaviate password
    :param api_key: weaviate api key, user should choose input either username/pwd or api_key
    :param index_path: vector database(FAISS, Chroma) index path, path to store database
    :param percent_fpf: percent of randomly selected buckets in FPF
    :param seed: random seed
    '''

    self.index_name = index_name
    self.rep_index_name = self.index_name + '__representatives'
    self.blob_ids = blob_ids
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
                                                                self.blob_ids.values.reshape(1, -1)[0])
    #TODO: load rep from stored database or parameter
    self.reps = None

  # TODO: Add memory efficient FPF Random Bucketter
  def _FPF(self, nb_buckets: Optional[int] = None):
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

    if self.reps is not None:
      self.reps = np.unique(np.concatenate((self.reps, reps)))
    else:
      self.reps = np.unique(reps)


  def get_representative_blob_ids(self) -> pd.DataFrame:
    '''
    get cluster representatives blob ids
    '''
    if self.reps is None:
      self._FPF()
    return self.blob_ids.iloc[self.reps]


  def get_topk_representatives_for_all(self, top_k: int = 5) -> pd.DataFrame:
    '''
    get topk representatives and distances for all blob index
    '''
    if self.reps is None:
      self._FPF()

    topk_reps, topk_dists = self.vector_database.execute(self.rep_index_name, self.embeddings, self.reps, top_k)
    topk_reps = self.blob_ids.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=self.blob_ids.squeeze())


  def get_topk_representatives_for_new_embeddings(
      self,
      new_blob_ids: pd.DataFrame,
      top_k: int = 5
  ) -> pd.DataFrame:
    '''
    get topk representatives and distances for new embeddings using stale cluster representatives,
    in other words, we don't need to use FPF to reselect cluster representatives
    '''
    new_embeddings = self.vector_database.get_embeddings_by_id(self.index_name,
                                                               new_blob_ids.values.reshape(1, -1)[0],
                                                               reload=True)
    if self.do_filter:
      topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name,
                                                                      new_embeddings,
                                                                      top_k,
                                                                      filter_ids=self.reps)
    else:
      topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name, new_embeddings, top_k)
    topk_reps = self.blob_ids.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=new_blob_ids.squeeze())


  def update_topk_representatives_for_all(
      self,
      new_blob_ids: pd.DataFrame,
      top_k: int = 5,
      nb_buckets: Optional[int] = None
  ) -> pd.DataFrame:
    '''
    when new embeddings are added, we update cluster representative ids and dists for all blob index
    '''
    #TODO: do we need to check if there is override bewteen blob_ids and new_blob_ids?
    self.blob_ids = pd.concat([self.blob_ids, new_blob_ids])
    new_embeddings = self.vector_database.get_embeddings_by_id(self.index_name,
                                                               new_blob_ids.values.reshape(1, -1)[0], reload=True)
    self.embeddings = np.concatenate((self.embeddings, new_embeddings), axis=0)

    self._FPF(nb_buckets)
    return self.get_topk_representatives_for_all(top_k)