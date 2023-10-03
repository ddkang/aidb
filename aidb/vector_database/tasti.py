import pandas as pd
from numba import njit, prange
from tqdm import tqdm
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDatabase
from typing import Optional
import numpy as np
from dataclasses import dataclass
from aidb.config.config_types import TastiConfig, VectorDatabaseType

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


@dataclass
class Tasti(TastiConfig):
  def __post_init__(self):
    self.rep_index_name = f"{self.index_name}__representatives"
    self.rand = np.random.RandomState(self.seed)
    self.do_filter = False
    self.initialize_vector_database()
    self.validate_index_name()
    self.initialize_embeddings()


  def initialize_vector_database(self):
    if self.vector_database_name == VectorDatabaseType.FAISS.value:
      self.initialize_faiss()
    elif self.vector_database_name == VectorDatabaseType.CHROMA.value:
      self.initialize_chroma()
    elif self.vector_database_name == VectorDatabaseType.WEAVIATE.value:
      self.initialize_weaviate()
    else:
      raise ValueError(f"{self.vector_database_name} is not supported, please use FAISS, Chroma, or Weaviate")


  def initialize_faiss(self):
    if self.index_path is None:
      raise ValueError('FAISS requires index path')
    self.vector_database = FaissVectorDatabase(self.index_path)
    self.vector_database.load_index(self.index_name)
    self.do_filter = True


  def initialize_chroma(self):
    if self.index_path is None:
      raise ValueError('Chroma requires index path')
    self.vector_database = ChromaVectorDatabase(self.index_path)


  def initialize_weaviate(self):
    if self.weaviate_auth.url is None:
      raise ValueError('Weaviate requires URL to connect')
    self.vector_database = WeaviateVectorDatabase(self.weaviate_auth)


  def validate_index_name(self):
    if self.index_name not in self.vector_database.index_list:
      raise ValueError(f"Index {self.index_name} doesn't exist in vector database")


  def initialize_embeddings(self):
    self.embeddings = self.vector_database.get_embeddings_by_id(self.index_name, self.blob_ids.values.reshape(1, -1)[0])


  # # TODO: Add memory efficient FPF Random Bucketter
  def _FPF(self, nb_buckets: Optional[int] = None):
    '''
    FPF mining algorithm and return cluster representative ids, old cluster representatives will always be kept
    '''
    if nb_buckets is not None:
      buckets = nb_buckets
    else:
      buckets = self.nb_buckets

    reps = np.full(buckets, -1)
    min_dists = np.full(len(self.embeddings), np.Inf, dtype=np.float32)
    num_random = int((1 - self.percent_fpf) * len(reps))
    random_reps = self.rand.choice(len(self.embeddings), num_random, replace=False)

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
    new_embeddings = self.vector_database.get_embeddings_by_id(self.index_name, new_blob_ids.values.reshape(1, -1)[0],
                                                               reload=True)
    if self.do_filter:
      topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name, new_embeddings,
                                                                      top_k, filter_ids=self.reps)
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