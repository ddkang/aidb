from enum import Enum
import numpy as np
import pandas as pd
from typing import Optional

from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateAuth, WeaviateVectorDatabase
from aidb.vector_database.tasti import Tasti

class VectorDatabaseType(Enum):
  FAISS = 'FAISS'
  CHROMA = 'Chroma'
  WEAVIATE = 'Weaviate'

class TastiTests():
  def __init__(
    self,
    index_name: str,
    data_size: int,
    embedding_dim: int,
    nb_buckets: int,
    vector_database_type: str = 'FAISS',
    percent_fpf: float = 0.75,
    seed: int = 1234,
    weaviate_auth: Optional[WeaviateAuth] = None,
    index_path: Optional[str] = None,
  ):
    self.index_name = index_name
    self.embedding_dim = embedding_dim
    self.nb_buckets = nb_buckets
    self.total_data = 0
    self.vd_type = vector_database_type
    self.seed = seed

    if vector_database_type == VectorDatabaseType.FAISS.value:
      vector_database = FaissVectorDatabase(index_path)
      vector_database.load_index(self.index_name)
    elif vector_database_type == VectorDatabaseType.CHROMA.value:
      vector_database = ChromaVectorDatabase(index_path)
    elif vector_database_type == VectorDatabaseType.WEAVIATE.value:
      vector_database = WeaviateVectorDatabase(weaviate_auth)
    else:
      raise ValueError(f"{vector_database_type} is not supported, please use FAISS, Chroma, or Weaviate")

    self.data = self.generate_data(data_size, embedding_dim)
    blob_ids = self.generate_blob_ids(data_size, 0)
    self.user_database = self.simulate_user_providing_database(index_path, weaviate_auth)

    self.vector_database = Tasti(index_name, blob_ids, vector_database, nb_buckets, percent_fpf, seed)


  def generate_data(self, data_size, emb_size):
    np.random.seed(self.seed)
    embeddings = np.random.rand(data_size, emb_size)
    data = pd.DataFrame({'id': range(self.total_data, self.total_data + data_size), 'values': embeddings.tolist()})
    self.total_data += data_size
    return data

  def generate_blob_ids(self, data_size, start):
    return pd.DataFrame({'id': range(start, start + data_size)})

  def simulate_user_providing_database(self, index_path: Optional[str], weaviate_auth: Optional[WeaviateAuth]):
    '''
    Originally, user will provide a vector database, and Tasti will read from it.
    This function is used to create a vector database to store original data
    '''
    user_database = None
    if self.vd_type == VectorDatabaseType.FAISS.value:
      user_database = FaissVectorDatabase(index_path)
      user_database.create_index(self.index_name, self.embedding_dim, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)
      user_database.save_index(self.index_name)

    elif self.vd_type == VectorDatabaseType.CHROMA.value:
      user_database = ChromaVectorDatabase(index_path)
      user_database.create_index(self.index_name, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)

    elif self.vd_type == VectorDatabaseType.WEAVIATE.value:
      user_database = WeaviateVectorDatabase(weaviate_auth)
      user_database.create_index(self.index_name, recreate_index=True)
      user_database.insert_data(self.index_name, self.data)

    return user_database


  def simulate_user_inserting_new_data(self, data_size):
    new_data = self.generate_data(data_size, self.embedding_dim)
    self.user_database.insert_data(self.index_name, new_data)
    if self.vd_type == 'FAISS':
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
    if self.vd_type == VectorDatabaseType.FAISS.value:
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
