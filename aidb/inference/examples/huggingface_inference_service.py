from typing import Dict, Union
import os
import requests
import pandas as pd

from aidb.inference.http_inference_service import HTTPInferenceService


class HuggingFaceNLP(HTTPInferenceService):
  def __init__(
      self,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      model: str=None):
    if token is None:
      token = os.environ['HF_API_KEY']
    super().__init__(
      name='huggingface_nlp',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class HuggingFaceVisionAudio(HTTPInferenceService):
  def __init__(
      self,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      model: str=None):
    if token is None:
      token = os.environ['HF_API_KEY']
    super().__init__(
      name='huggingface_nlp',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      copy_input=False,
      batch_supported=False,
      is_single=False,
      response_keys_to_columns=response_keys_to_columns,
    )


  def convert_input_to_request(self, input: pd.Series) -> Dict:
    return input.to_dict()


  def request(self, request: Dict) -> Dict:
    with open(request['filename'], 'rb') as f:
      response = requests.post(self._url, data=f, headers=self._headers)
    response.raise_for_status()
    return response.json()
