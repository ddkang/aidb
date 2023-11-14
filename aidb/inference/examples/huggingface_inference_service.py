from typing import Dict, Union, List
import os
import requests
import pandas as pd

from aidb.inference.http_inference_service import HTTPInferenceService


class HuggingFaceNLP(HTTPInferenceService):
  def __init__(
      self,
      name: str='huggingface_nlp',
      is_single: bool=False,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      rate_limit: Union[int, None]=None,
      input_columns_types: Union[List, None]=None,
      output_columns_types: Union[List, None]=None,
      model: str=None):
    if token is None:
      token = os.environ['HF_API_KEY']
    super().__init__(
      name=name,
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      batch_supported=False,
      is_single=is_single,
      rate_limit=rate_limit,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
      input_columns_types=input_columns_types,
      output_columns_types=output_columns_types,
    )


class HuggingFaceVisionAudio(HTTPInferenceService):
  def __init__(
      self,
      name: str='huggingface_vision_audio',
      is_single: bool=False,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      rate_limit: Union[int, None]=None,
      output_columns_types: Union[List, None]=None,
      model: str=None):
    if token is None:
      token = os.environ['HF_API_KEY']
    super().__init__(
      name=name,
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      batch_supported=False,
      is_single=is_single,
      rate_limit=rate_limit,
      response_keys_to_columns=response_keys_to_columns,
      output_columns_types=output_columns_types,
    )


  def convert_input_to_request(self, input: pd.Series) -> Dict:
    return input.to_dict().values()[0]


  def request(self, request: str) -> Dict:
    with open(request, 'rb') as f:
      response = requests.post(self._url, data=f, headers=self._headers)
    response.raise_for_status()
    return response.json()
