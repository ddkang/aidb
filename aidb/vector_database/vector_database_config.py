from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional

from aidb.vector_database.vector_database import VectorDatabase


@dataclass
class TastiConfig:
  '''
  :param index_name: vector database index name
  :param blob_ids: blob index in blob table, it should be unique for each data record
  :param nb_buckets: number of buckets for FPF, it should be same as the number of buckets for oracle
  :param vector_database: vector database type, it should be FAISS, Chroma or Weaviate
  :param percent_fpf: percent of randomly selected buckets in FPF
  :param seed: random seed
  :param weaviate_auth: Weaviate authentification
  :param index_path: vector database(FAISS, Chroma) index path, path to store database
  :param reps: representative ids
  '''
  index_name: str
  blob_ids: pd.DataFrame
  vector_database: VectorDatabase
  nb_buckets: int
  percent_fpf: float = 0.75
  seed: int = 1234
  reps: Optional[np.ndarray] = field(default=None)

  def __post_init__(self):
    pass