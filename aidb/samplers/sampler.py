import abc
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SampledBlobId:
  weight: float
  mass: float

@dataclass
class SampledBlob(SampledBlobId):
  statistic: List[float]
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
