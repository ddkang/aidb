import numpy as np
import pandas as pd
from typing import Union, NewType, Optional

from aidb.utils.constants import BLOB_TABLE_NAMES_TABLE, table_name_for_rep_and_topk_and_blob_mapping
from aidb.utils.db import create_sql_engine, infer_dialect
from aidb.vector_database.chroma_vector_database import ChromaVectorDatabase
from aidb.vector_database.faiss_vector_database import FaissVectorDatabase
from aidb.vector_database.weaviate_vector_database import WeaviateAuth, WeaviateVectorDatabase

IndexPath = NewType('index_path', str)

class VectorDatabaseSetup:
  def __init__(
      self,
      connection_uri: str,
      blob_table_name: str,
      vd_type: str,
      index_name: str,
      auth: Union[IndexPath, WeaviateAuth]
  ):
    self._dialect = infer_dialect(connection_uri)
    self._sql_engine = create_sql_engine(connection_uri)
    self.blob_table_name = blob_table_name
    self.vd_type = vd_type.upper()
    self.index_name = index_name
    self.auth = auth


  async def _retrieve_blob_keys(self, conn):

    query_str = f'''
                    SELECT blob_key
                    FROM {BLOB_TABLE_NAMES_TABLE} 
                    WHERE table_name = '{self.blob_table_name}';
                 '''

    df = await conn.run_sync(lambda conn: pd.read_sql(query_str, conn))

    blob_keys = df['blob_key'].to_list()
    return blob_keys


  def _get_embedding(self, vector_database_df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function is used to generate embedding based on blob file, the input has foramt
    pd.DataFrame({'vector_id':[], 'path':[]}, the output should have format pd.DataFrame({'id':[], 'values':[]}
    '''

    # FIXME: this should be replaced real generate_embedding function
    def generate_data(vector_database_df):
      np.random.seed(1234)
      embeddings = np.random.rand(len(vector_database_df), 128)
      data = pd.DataFrame({'id': vector_database_df['vector_id'], 'values': embeddings.tolist()})
      return data

    return generate_data(vector_database_df)


  def _create_vector_database_index(self, data: pd.DataFrame):
    vector_database = {
      'FAISS': FaissVectorDatabase,
      'CHROMA': ChromaVectorDatabase,
      'WEAVIATE': WeaviateVectorDatabase
    }

    try:
      user_vector_database = vector_database[self.vd_type](self.auth)
    except KeyError:
      raise ValueError(f'{self.vd_type} is not a supported database type. We support FAISS, Chroma and Weaviate.')

    if self.vd_type == 'FAISS':
      embedding_length = len(data['values'].iloc[0])
      user_vector_database.create_index(self.index_name, embedding_length, recreate_index=True)
    else:
      user_vector_database.create_index(self.index_name, recreate_index=True)

    user_vector_database.insert_data(self.index_name, data)


  async def setup(self, path_column: Optional[str] = None):
    if path_column is None:
      path_column = 'path'

    async with self._sql_engine.begin() as conn:
      select_columns = await self._retrieve_blob_keys(conn)
      select_columns.append(path_column)
      select_str = ', '.join(select_columns)
      query_str = f'''
                      SELECT {select_str}
                      FROM {self.blob_table_name}
                   '''

      df = await conn.run_sync(lambda conn: pd.read_sql(query_str, conn))

      df['vector_id'] = list(range(len(df)))
      blob_mapping_table_df = df.drop(columns=path_column)
      # create blob mapping table
      _, _, blob_mapping_table_name = table_name_for_rep_and_topk_and_blob_mapping([self.blob_table_name])
      await conn.run_sync(lambda conn: blob_mapping_table_df.to_sql(blob_mapping_table_name, conn,
                                                                    index=False, if_exists='append'))

    vector_database_df = df[['vector_id', path_column]]
    embeddings_df = self._get_embedding(vector_database_df)

    self._create_vector_database_index(embeddings_df)
