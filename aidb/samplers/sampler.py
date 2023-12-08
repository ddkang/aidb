import abc
import copy
from dataclasses import dataclass
from typing import List


@dataclass
class SampledBlobId:
  weight: float
  mass: float

@dataclass
class SampledBlob(SampledBlobId):
  statistics: List[float]
  num_items: int


  def get_specific_data_by_index(self, index):
    new_sampled_blob = copy.copy(self)
    if 0 <= index < len(self.statistics):
      new_sampled_blob.statistics = [self.statistics[index]]
    else:
      raise Exception('Index exceed the length of result list')

    return new_sampled_blob


class Sampler(abc.ABC):
  @abc.abstractmethod
  def reset(self):
    pass

  @abc.abstractmethod
  def sample(self, num_samples: int) -> list:
    pass

  @abc.abstractmethod
  def sample_next_n(self, num_samples: int) -> list:
    pass
