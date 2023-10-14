from dataclasses import dataclass


@dataclass
class SampledBlob():
  weight: float
  mass: float
  statistic: float
  num_items: int