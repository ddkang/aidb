import numpy as np
import pandas as pd
from typing import Optional

from aidb.config.config_types import WeaviateAuth, VectorDatabaseType
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateVectorDatabase
from aidb.vector_database.tasti import Tasti


class TastiTests():
  def __init__(
    self,
    index_name: str,
    data_size: int,
    embedding_dim: int,
    nb_buckets: int,
    vector_database: str = 'FAISS',
    percent_fpf: float = 0.75,
    seed: int = 1234,
    weaviate_auth: Optional[WeaviateAuth] = None,
    index_path: Optional[str] = None,
  ):
    #
    self.index_name = index_name
    self.embedding_dim = embedding_dim
    self.vd_name = vector_database
    self.index_path = index_path
    self.weaviate_auth = weaviate_auth
    self.nb_buckets = nb_buckets
    self.seed = seed
    self.total_data = 0

    self.data = self.generate_data(data_size, embedding_dim)
    blob_ids = self.generate_blob_ids(data_size, 0)
    self.user_database = self.simulate_user_providing_database()
    self.vector_database = Tasti(index_name, blob_ids, nb_buckets, self.vd_name,
                                 percent_fpf, seed, weaviate_auth, index_path)



  def generate_data(self, data_size, emb_size, seed: Optional[int] = None):
    if seed:
      np.random.seed(seed)
    else:
      np.random.seed(self.seed)

    embeddings = np.random.rand(data_size, emb_size)
    data = pd.DataFrame({'id': range(self.total_data, self.total_data + data_size), 'values': embeddings.tolist()})
    self.total_data += data_size
    return data

  def generate_blob_ids(self, data_size, start):
    return pd.DataFrame({'id': range(start, start + data_size)})

  def simulate_user_providing_database(self):
    '''
    Originally, user will provide a vector database, and Tasti will read from it.
    This function is used to create a vector database to store original data
    '''
    user_database = None
    if self.vd_name == VectorDatabaseType.FAISS.value:
      user_database = FaissVectorDatabase(self.index_path)
      user_database.create_index(self.index_name, self.embedding_dim, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)
      user_database.save_index(self.index_name)

    elif self.vd_name == VectorDatabaseType.CHROMA.value:
      user_database = ChromaVectorDatabase(self.index_path)
      user_database.create_index(self.index_name, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)

    elif self.vd_name == VectorDatabaseType.WEAVIATE.value:
      user_database = WeaviateVectorDatabase(self.weaviate_auth)
      user_database.create_index(self.index_name, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)

    return user_database


  def simulate_user_inserting_new_data(self, data_size):
    new_data = self.generate_data(data_size, self.embedding_dim, 12345)
    self.user_database.insert_data(self.index_name, new_data)
    if self.vd_name == 'FAISS':
      self.user_database.save_index(self.index_name)


  def test(self):
    representative_blob_ids = self.vector_database.get_representative_blob_ids()
    print('The shape of cluster representative ids', representative_blob_ids.shape)
    #get culster representatives ids
    print(representative_blob_ids)
    topk_representatives = self.vector_database.get_topk_representatives_for_all()
    #get topk representatives and dists for all data
    print(topk_representatives)


    #Chroma uses HNSW, which will not return exact search result
    if self.vd_name == VectorDatabaseType.FAISS.value:
      for representative_id in list(representative_blob_ids['id']):
        assert representative_id in topk_representatives.loc[representative_id]['topk_reps']

    self.simulate_user_inserting_new_data(10000)
    new_blob_ids = self.generate_blob_ids(10000, 10000)
    # get topk representatives and dists for new data based on stale representatives
    print(self.vector_database.get_topk_representatives_for_new_embeddings(new_blob_ids))
    # reselect cluster representatives, recompute topk representatives and dists for all data
    print(self.vector_database.update_topk_representatives_for_all(new_blob_ids))
    # We can see the old cluster representative is kept
    print('The total number of cluster representatives is:', len(self.vector_database.reps))

def test(vector_database):
  index_name = 'Tasti'
  data_size = 10000
  embedding_dim = 128
  nb_buckets = 1000
  url = ''
  api_key = ''
  index_path = './'
  weaviate_auth = WeaviateAuth(url, api_key=api_key)
  tasti_test = TastiTests(index_name, data_size, embedding_dim, nb_buckets, vector_database,
                          weaviate_auth=weaviate_auth, index_path=index_path)
  tasti_test.test()

if __name__ == '__main__':
    test('FAISS')
    # test('Chroma')
    #too slow
    # test('Weaviate')
