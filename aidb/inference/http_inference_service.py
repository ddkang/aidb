from flatten_json import unflatten_list
from typing import Dict, List, Tuple, Union

import pandas as pd
import requests

from aidb.config.config_types import AIDBListType
from aidb.inference.cached_inference_service import CachedInferenceService
from aidb.utils.perf_utils import call_counter


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
      batch_supported: bool=False,
      columns_to_input_keys: List[Union[str, tuple]]=None,
      input_columns_types: Union[List, None]=None,
      response_keys_to_columns: List[Union[str, tuple]]=None,
      output_columns_types: Union[List, None]=None,
      **kwargs
  ):
    '''
    :param str url: The URL to send the request to. The request will be a POST request.
    :param dict headers: The headers to send with the request.
    :param bool batch_supported: Whether the server supports batch requests.
    '''
    super().__init__(*args, **kwargs)
    self._url = url
    self._headers = headers
    self._default_args = default_args
    self._batch_supported = batch_supported
    self._columns_to_input_keys = columns_to_input_keys
    self._response_keys_to_columns = response_keys_to_columns

    assert not input_columns_types or len(input_columns_types) == len(columns_to_input_keys), \
      f'input_columns_types must be None or have same length as columns_to_input_keys'
    self._input_columns_types = input_columns_types
    assert not output_columns_types or len(output_columns_types) == len(response_keys_to_columns), \
      f'output_columns_types must be None or have same length as response_keys_to_columns'
    self._output_columns_types = output_columns_types

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

    if self._input_columns_types is not None:
      for k, _type in zip(dict_input_keys, self._input_columns_types):
        if len(dict_input[k]) > 0:
          assert isinstance(dict_input[k][0], _type), f'Input column {k} must be of type {_type}'

    columns_to_input_keys = self._columns_to_input_keys.copy()
    if self._default_args is not None:
      idx = len(columns_to_input_keys)
      for k, v in self._default_args.items():
        if k not in dict_input_keys:
          dict_input_keys.insert(idx, k)
          dict_input[k] = [v] * num_rows
          columns_to_input_keys.append(k)
          idx += 1

    # to support arbitrary batch size
    # assume all numerical index form lists
    for k, v in enumerate(columns_to_input_keys):
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
    print(request)
    return request


  def request(self, request: Dict) -> Dict:
    response = requests.post(self._url, json=request, headers=self._headers)
    response.raise_for_status()
    return response.json()


  def convert_response_to_output(self, response: Dict) -> pd.DataFrame:
    self._response_keys_to_columns = {k: i for i, k in enumerate(self._response_keys_to_columns)}
    output = convert_response_to_output(response, self._response_keys_to_columns)
    if not any(isinstance(value, list) for value in output.values()):
      output = {k: [v] for k, v in output.items()}

    if self._output_columns_types is not None:
      for k, _type in zip(output.keys(), self._output_columns_types):
        if len(output[k]) > 0:
          assert isinstance(output[k][0], _type), f'Output column {k} must be of type {_type}'

    return pd.DataFrame(output)


  @call_counter
  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    request = self.convert_input_to_request(input)
    response = self.request(request)
    output = self.convert_response_to_output(response)

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

    return outputs
