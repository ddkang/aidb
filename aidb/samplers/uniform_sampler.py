import numpy as np
import pandas as pd
from typing import List

from aidb.samplers.sampler import SampledBlobId, Sampler


class UniformBlobSampler(Sampler):
  def __init__(
    self,
    blob_id_count: int,
    seed: int = None
  ):
    super().__init__()
    self._blob_ids = pd.DataFrame({'blob_id': list(range(blob_id_count))})
    self.reset(seed)


  def reset(self, seed: int = None):
    self._total_sampled = 0
    self._rand = np.random.RandomState(seed)
    self._shuffled_blob_ids = self._blob_ids.sample(frac=1, random_state=self._rand).reset_index(drop=True)


  def remaining_samples(self) -> int:
    return len(self._shuffled_blob_ids) - self._total_sampled


  def sample(self, _) -> list:
    raise Exception('Do not call UniformBlobSampler.sample directly')


  def sample_next_n(self, num_samples: int) -> List[SampledBlobId]:
    if self._total_sampled + num_samples > len(self._shuffled_blob_ids):
      raise Exception('Not enough blob ids to sample')

    #return num_samples start from total_sampled
    sampled_blob_ids = self._shuffled_blob_ids[self._total_sampled:][:num_samples]

    samples = []
    for idx in range(num_samples):
      samples.append(
        SampledBlobId(sampled_blob_ids.iloc[idx], 1. / len(self._blob_ids), 1.)
      )

    self._total_sampled += num_samples
    return samples
