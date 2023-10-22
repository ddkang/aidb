from flatten_json import unflatten_list
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests

from aidb.config.config_types import AIDBListType
from aidb.inference.cached_inference_service import CachedInferenceService


class HTTPInferenceService(CachedInferenceService):
  def __init__(
      self,
      *args,
      url: str=None,
      headers: Union[Dict, None]=None,
      default_args: Union[Dict, None]=None,
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
    self._default_args = default_args
    self._copy_input = copy_input
    self._batch_supported = batch_supported
    self._columns_to_input_keys = columns_to_input_keys
    self._response_keys_to_columns = response_keys_to_columns
    self._separator = '~'


  def signature(self) -> Tuple[List, List]:
    raise NotImplementedError()
  

  def convert_input_to_request(self, input: pd.Series) -> Dict:
    request = {}
    for k, v in input.to_dict().items():
      if k in self._columns_to_input_keys:
        key = self._columns_to_input_keys[k]
        key = self._separator.join(key) if isinstance(key, tuple) else key
        request[key] = v
    if self._default_args is not None:
      for k, v in self._default_args.items():
        if k not in request:
          request[k] = v
    return unflatten_list(request, self._separator)


  def request(self, request: Dict) -> Dict:
    response = requests.post(self._url, json=request, headers=self._headers)
    response.raise_for_status()
    return response.json()


  def convert_response_to_output(self, response: Dict) -> pd.DataFrame:
    # some response may be a list of indefinite length
    # but users may want to specify the maximum via _response_keys_to_columns
    output = {v: [] for v in self._response_keys_to_columns.values()}
    for k, v in self._response_keys_to_columns.items():
      if isinstance(k, str):
        k = (k,)
      retrieved = True
      response_copy = response.copy()
      for key in k:
        if isinstance(key, AIDBListType):
          if isinstance(response_copy, list):
            continue
          else:
            retrieved = False
            break
        else: # isinstance(key, str)
          if isinstance(key, str) and key.isnumeric(): # single instance in list
            key = int(key)
          if isinstance(response_copy, dict):
            if key in response_copy:
              response_copy = response_copy[key]
            else:
              retrieved = False
              break
          else: # isinstance(response_copy, list)
            new_response_copy = []
            for r in response_copy:
              if key in r:
                new_response_copy.append(r[key])
            if len(new_response_copy) == 0:
              retrieved = False
              break
            else:
              response_copy = new_response_copy

      if retrieved:
        output[v] = response_copy
    if not any(isinstance(value, list) for value in output.values()):
      output = {k: [v] for k, v in output.items()}
    print(output)
    return pd.DataFrame(output)



  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    request = self.convert_input_to_request(input)
    response = self.request(request)
    output = self.convert_response_to_output(response)

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
