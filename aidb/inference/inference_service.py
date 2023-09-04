import abc
from dataclasses import dataclass
from typing import List, Tuple, Union

import pandas as pd


@dataclass
class InferenceService(abc.ABC):
  '''
  The inference service is a wrapper around calling external ML models. For the
  inference service, we enforce that the input is a single pandas row (Series).
  If inputs span multiple tables, the caller will join the data into the Series.
  The output is a single pandas DataFrame with zero or more rows.

  The batch inference also takes in a single pandas DataFrame, but with multiple
  rows. The output is a list of DataFrames.

  '''
  name: str
  is_single: bool  # Return a single value or a list of values
  cost: Union[float, None] = None
  preferred_batch_size: int = 1


  @abc.abstractproperty
  def signature(self) -> Tuple[List, List]:
    pass


  @abc.abstractmethod
  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    pass


  def infer_batch(self, inputs: pd.DataFrame) -> List[pd.DataFrame]:
    return [self.infer_one(row) for _, row in inputs.iterrows()]


  async def infer_one_async(self, input: pd.Series) -> pd.DataFrame:
    return self.infer_one(input)


  async def infer_batch_async(self, inputs: pd.DataFrame) -> List[pd.DataFrame]:
    return self.infer_batch(inputs)