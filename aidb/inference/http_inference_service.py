from flatten_json import unflatten_list
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests

from aidb.config.config_types import AIDBListType
from aidb.inference.cached_inference_service import CachedInferenceService


def convert_response_to_output(
    response: Union[Dict, List],
    _response_keys_to_columns: Dict[Union[str, tuple], int]) -> Dict:
  output = {v: [] for v in _response_keys_to_columns.values()}
  for k, v in _response_keys_to_columns.items():
    if not isinstance(k, tuple):
      k = (k,)
    response_copy = response.copy()
    for idx, key in enumerate(k):
      if isinstance(key, AIDBListType) and isinstance(response_copy, list):
        if idx != len(k) - 1:
          new_response_copy = []
          for r in response_copy:
            new_key = k[(idx+1):]
            current_response = convert_response_to_output(r, {new_key: new_key})
            if current_response is not None and new_key in current_response:
              if isinstance(current_response[new_key], list): 
                new_response_copy.extend(current_response[new_key])
              else:
                new_response_copy.append(current_response[new_key])
          response_copy = new_response_copy if len(new_response_copy) > 0 else None
          break
      elif (isinstance(key, int) and \
           (isinstance(response_copy, list) and key < len(response_copy)) or \
           (isinstance(response_copy, dict) and key in response_copy)) or \
           (isinstance(key, str) and isinstance(response_copy, dict) and key in response_copy):
        response_copy = response_copy[key]
      else:
        response_copy = None
        break

    if response_copy is not None:
      output[v] = response_copy
  return output


class HTTPInferenceService(CachedInferenceService):
  def __init__(
      self,
      *args,
      url: str=None,
      headers: Union[Dict, None]=None,
      default_args: Union[Dict, None]=None,
      copy_input: bool=True,
      batch_supported: bool=False,
      columns_to_input_keys: Dict[int, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], int]=None,
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
  

  def convert_input_to_request(self, input: Union[pd.Series, pd.DataFrame]) -> Dict:
    request = {}
    remove_ghost_key = False
    if isinstance(input, pd.Series):
      num_rows = 1
      dict_input = {k: [v] for k, v in input.to_dict().items()}
    else: # isinstance(input, pd.DataFrame)
      num_rows = len(input)
      dict_input = input.to_dict(orient='list')
    dict_input_keys = list(dict_input.keys())
    # to support arbitrary batch size
    # assume all numerical index form lists
    for k, v in self._columns_to_input_keys.items():
      if k > len(dict_input_keys):
        continue
      k = dict_input_keys[k]
      if isinstance(v, tuple):
        aidb_list_count = sum(1 for e in v if isinstance(e, AIDBListType))
        if aidb_list_count == 0:
          key = tuple(str(_k) for _k in v)
          key = self._separator.join(key)
          request[key] = dict_input[k][0]
        elif aidb_list_count == 1:
          for i in range(num_rows):
            key = tuple(f'{i}' if isinstance(_k, AIDBListType) else f'{_k}' for _k in v)
            if isinstance(v[0], AIDBListType):
              key = ('_', ) + key # all converted keys should start with AIDBListType
              remove_ghost_key = True
            key = self._separator.join(key)
            request[key] = dict_input[k][i]
        else:
          raise ValueError(f'Cannot have more than 1 AIDBListType in columns_to_input_keys')
      else: # isinstance(v, str)
        request[v] = dict_input[k][0]
    request = unflatten_list(request, self._separator)
    if remove_ghost_key:
      request = request['_']
    if self._default_args is not None:
      if isinstance(request, dict):
        for k, v in self._default_args.items():
          if k not in request:
            request[k] = v
      else: # isinstance(request, list)
        for r in request:
          if isinstance(r, dict):
            for k, v in self._default_args.items():
              if k not in r:
                r[k] = v
    return request


  def request(self, request: Dict) -> Dict:
    response = requests.post(self._url, json=request, headers=self._headers)
    response.raise_for_status()
    return response.json()


  def convert_response_to_output(self, response: Dict) -> pd.DataFrame:
    output = convert_response_to_output(response, self._response_keys_to_columns)
    if not any(isinstance(value, list) for value in output.values()):
      output = {k: [v] for k, v in output.items()}
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
