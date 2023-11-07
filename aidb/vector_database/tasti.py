import numpy as np
import pandas as pd
from numba import njit, prange
from typing import Optional

from aidb.vector_database.vector_database_config import TastiConfig

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


class Tasti(TastiConfig):
  def __post_init__(self):
    self.rep_index_name = f'{self.index_name}__representatives'
    self.vector_ids = None
    self.reps = None
    self.rand = np.random.RandomState(self.seed)


  def set_vector_ids(self, vector_ids: pd.DataFrame):
    self.vector_ids = vector_ids
    self.embeddings = self.vector_database.get_embeddings_by_id(
      self.index_name,
      self.vector_ids.values.reshape(1, -1)[0],
      reload=True
    )


  def set_existing_reps(self, reps: np.ndarray):
    self.reps = reps


  # # TODO: Add memory efficient FPF Random Bucketter
  def _FPF(self, nb_buckets: Optional[int] = None):
    '''
    FPF mining algorithm and return cluster representative ids, old cluster representatives will always be kept
    '''
    if nb_buckets is not None:
      buckets = nb_buckets
    else:
      buckets = self.nb_buckets

    if self.vector_ids is None:
      raise Exception('Vector_ids is None, please set it first.')

    reps = np.full(buckets, -1)
    min_dists = np.full(len(self.embeddings), np.Inf, dtype=np.float32)
    num_random = int((1 - self.percent_fpf) * len(reps))
    random_reps = self.rand.choice(len(self.embeddings), num_random, replace=False)

    reps[0] = random_reps[0]
    get_and_update_dists(self.embeddings[reps[0]], self.embeddings, min_dists)

    for i in range(1, num_random):
      reps[i] = random_reps[i]
      get_and_update_dists(self.embeddings[reps[i]], self.embeddings, min_dists)

    for i in range(num_random, buckets):
      reps[i] = np.argmax(min_dists)
      get_and_update_dists(self.embeddings[reps[i]], self.embeddings, min_dists)

    if self.reps is not None:
      self.reps = np.unique(np.concatenate((self.reps, reps)))
    else:
      self.reps = np.unique(reps)


  def get_representative_vector_ids(self) -> pd.DataFrame:
    '''
    get cluster representatives blob ids
    '''
    if self.reps is None:
      self._FPF()
    rep_id = self.vector_ids.iloc[self.reps]
    rep_id.set_index('vector_id', inplace=True, drop=True)
    return rep_id


  def get_topk_representatives_for_all(self, top_k: int = 5) -> pd.DataFrame:
    '''
    get topk representatives and distances for all blob index
    '''
    if self.reps is None:
      self._FPF()
    topk_reps, topk_dists = self.vector_database.execute(self.rep_index_name, self.embeddings, self.reps, top_k)
    topk_reps = self.vector_ids.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=self.vector_ids.squeeze())


  def get_topk_representatives_for_new_embeddings(
      self,
      new_vector_ids: pd.DataFrame,
      top_k: int = 5
  ) -> pd.DataFrame:
    '''
    get topk representatives and distances for new embeddings using stale cluster representatives,
    in other words, we don't need to use FPF to reselect cluster representatives
    '''
    new_embeddings = self.vector_database.get_embeddings_by_id(
        self.index_name,
        new_vector_ids.values.reshape(1, -1)[0],
        reload=True
    )
    topk_reps, topk_dists = self.vector_database.query_by_embedding(self.rep_index_name, new_embeddings, top_k)
    topk_reps = self.vector_ids.iloc[np.concatenate(topk_reps)].values.reshape(-1, top_k)
    data = {'topk_reps': list(topk_reps), 'topk_dists': list(topk_dists)}
    return pd.DataFrame(data, index=new_vector_ids.squeeze())


  def update_topk_representatives_for_all(
      self,
      new_vector_ids: pd.DataFrame,
      top_k: int = 5,
      nb_buckets: Optional[int] = None
  ) -> pd.DataFrame:
    '''
    when new embeddings are added, we update cluster representative ids and dists for all blob index
    '''
    #TODO: do we need to check if there is override bewteen vector_ids and new_vector_ids?
    self.vector_ids = pd.concat([self.vector_ids, new_vector_ids])
    new_embeddings = self.vector_database.get_embeddings_by_id(
        self.index_name,
        new_vector_ids.values.reshape(1, -1)[0],
        reload=True
    )
    self.embeddings = np.concatenate((self.embeddings, new_embeddings), axis=0)

    self._FPF(nb_buckets)
    return self.get_topk_representatives_for_all(top_k)