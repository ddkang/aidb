import abc
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
