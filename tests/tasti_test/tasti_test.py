from enum import Enum
import numpy as np
import os
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
    vector_database_type: VectorDatabaseType,
    percent_fpf: float = 0.75,
    seed: int = 1234,
    weaviate_auth: Optional[WeaviateAuth] = None,
    index_path: Optional[str] = None,
  ):
    self.index_name = index_name
    self.data_size = data_size
    self.embedding_dim = embedding_dim
    self.nb_buckets = nb_buckets
    self.total_data = 0
    self.vd_type = vector_database_type
    self.seed = seed

    self.data, vector_ids = self.generate_data(self.data_size, embedding_dim)
    self.user_database = self.simulate_user_providing_database(index_path, weaviate_auth)

    self.tasti = Tasti(index_name, vector_ids, self.user_database, nb_buckets, percent_fpf, seed)


  def generate_data(self, data_size, emb_size):
    np.random.seed(self.seed)
    embeddings = np.random.rand(data_size, emb_size)
    data = pd.DataFrame({'id': range(self.total_data, self.total_data + data_size), 'values': embeddings.tolist()})
    vector_ids = pd.DataFrame({'id': range(self.total_data, self.total_data + data_size)})
    self.total_data += data_size
    return data, vector_ids


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
    new_data, blob_id = self.generate_data(data_size, self.embedding_dim)
    self.user_database.insert_data(self.index_name, new_data)
    if self.vd_type == 'FAISS':
      self.user_database.save_index(self.index_name)
    return new_data, blob_id


  def test(self):
    representative_vector_ids = self.tasti.get_representative_vector_ids()
    print('The shape of cluster representative ids', representative_vector_ids.shape)
    # get culster representatives ids
    print(representative_vector_ids)
    topk_representatives = self.tasti.get_topk_representatives_for_all()
    # get topk representatives and dists for all data
    print(topk_representatives)


    # Chroma uses HNSW, which will not return exact search result
    if self.vd_type == VectorDatabaseType.FAISS.value:
      for representative_id in list(representative_vector_ids):
        assert representative_id in topk_representatives.loc[representative_id]['topk_reps']

    new_data, new_vector_ids = self.simulate_user_inserting_new_data(self.data_size)
    # get topk representatives and dists for new data based on stale representatives
    print(self.tasti.get_topk_representatives_for_new_embeddings(new_vector_ids))
    # reselect cluster representatives, recompute topk representatives and dists for all data
    print(self.tasti.update_topk_representatives_for_all(new_vector_ids))
    # We can see the old cluster representative is kept
    print('The total number of cluster representatives is:', len(self.tasti.reps))


def test(
    index_name: str,
    vector_database_type: VectorDatabaseType,
    data_size: int,
    embedding_dim: int,
    nb_buckets: int,
    index_path: Optional[str] = None,
    weaviate_auth: Optional[WeaviateAuth] = None
):
  tasti_test = TastiTests(index_name, data_size, embedding_dim, nb_buckets, vector_database_type,
                          weaviate_auth=weaviate_auth, index_path=index_path)
  tasti_test.test()


if __name__ == '__main__':
    print(f'Running FAISS vector database')
    test('faiss', VectorDatabaseType.FAISS.value, data_size=10000,
         embedding_dim=128, nb_buckets=1000, index_path='./')

    print(f'Running Chroma vector database')
    test('chroma', VectorDatabaseType.CHROMA.value, data_size=10000,
         embedding_dim=128, nb_buckets=1000, index_path='./')

    # too slow
    print(f'Running Weaviate vector database')
    url = ''
    api_key = os.environ.get('WEAVIATE_API_KEY')
    weaviate_auth = WeaviateAuth(url, api_key=api_key)
    test('Weaviate', VectorDatabaseType.WEAVIATE.value, data_size=200,
         embedding_dim=128, nb_buckets=50, weaviate_auth=weaviate_auth)
