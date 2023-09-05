from typing import Dict, List, Tuple, Union

import pandas as pd
import requests

from aidb.inference.cached_inference_service import CachedInferenceService


class HTTPInferenceService(CachedInferenceService):
  def __init__(
      self,
      *args,
      url: str=None,
      headers: Union[Dict, None]=None,
      copy_input: bool=True,
      batch_supported: bool=False,
      **kwargs
  ):
    '''
    :param str url: The URL to send the request to. The request will be a POST request.
    :param dict headers: The headers to send with the request.
    :param bool copy_input: Whether to copy the input _after_ receiving the request from the server.
    :param bool batch_supported: Whether the server supports batch requests.
    '''
    super().__init__(*args, **kwargs)
    self._url = url
    self._headers = headers
    self._copy_input = copy_input
    self._batch_supported = batch_supported


  def signature(self) -> Tuple[List, List]:
    raise NotImplementedError()


  def infer_one(self, input: pd.Series):
    # TODO: index or columns?
    body = input.to_json(orient='columns')
    response = requests.post(self._url, data=body, headers=self._headers)
    response.raise_for_status()
    response = response.json()
    output = pd.DataFrame(response)
    # TODO: is this correct for zero or 2+ outputs?
    if self._copy_input:
      output = output.assign(**input)
    return output


  def infer_batch(self, inputs: pd.DataFrame) -> List[pd.DataFrame]:
    if not self._batch_supported:
      return super().infer_batch(inputs)
    
    body = inputs.to_json(orient='records')
    response = requests.post(self._url, data=body, headers=self._headers)
    response.raise_for_status()

    # We assume the server returns a list of responses
    response = response.json()
    outputs = [pd.read_json(r, orient='records') for r in response]
    if self._copy_input:
      outputs = [o.assign(**i) for o, (_, i) in zip(outputs, inputs.iterrows())]

    return outputs
