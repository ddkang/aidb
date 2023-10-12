from flatten_json import flatten, unflatten_list
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
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
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
    self._columns_to_input_keys = columns_to_input_keys
    self._response_keys_to_columns = response_keys_to_columns
    self.separator = '.'


  def signature(self) -> Tuple[List, List]:
    raise NotImplementedError()
  

  def request(self, input: pd.Series) -> Dict:
    request = {}
    for k, v in input.to_dict().items():
      if k in self._columns_to_input_keys:
        key = self._columns_to_input_keys[k]
        key = self.separator.join(key) if isinstance(key, tuple) else key
        request[key] = v
    request_unflatten = unflatten_list(request, self.separator)

    response = requests.post(self._url, json=request_unflatten, headers=self._headers)
    response.raise_for_status()
    return response.json()


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    response = self.request(input)

    # some response may be a list of indefinite length
    # but users may want to specify the maximum via _response_keys_to_columns
    response_is_list = isinstance(response, list)
    if response_is_list:
      response = {'_': response} # only hf returns a list
    response_flatten = flatten(response, self.separator)
    output = {}
    for k, v in response_flatten.items():
      k = tuple(k.split(self.separator))
      if response_is_list:
        k = k[1:] # remove '_' for list
      if k in self._response_keys_to_columns:
        output[self._response_keys_to_columns[k]] = v

    output = pd.DataFrame([output])
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
