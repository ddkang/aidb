import abc
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SampledBlobId:
  blob_id: Any
  weight: float
  mass: float

@dataclass
class SampledBlob(SampledBlobId):
  sample: Dict[str, pd.DataFrame] # Table -> df
  statistic: float
  num_items: Dict[str, int] # For set samplers, in case of multiple table outputs, table -> count