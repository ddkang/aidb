from typing import Dict, Union
import requests
import pandas as pd

from aidb.inference.http_inference_service import HTTPInferenceService


class HuggingFaceNLP(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      columns_to_input_keys: Dict[str, Union[str, tuple]],
      response_keys_to_columns: Dict[Union[str, tuple], str],
      model: str):
    '''
    :param str token: The token to use for authentication.
    :param str model: The model to use for inference.
    '''
    super().__init__(
      name='huggingface_nlp',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class HuggingFaceVisionAudio(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      response_keys_to_columns: Dict[Union[str, tuple], str],
      model: str):
    '''
    :param str token: The token to use for authentication.
    :param str model: The model to use for inference.
    '''
    super().__init__(
      name='huggingface_nlp',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
      response_keys_to_columns=response_keys_to_columns,
    )


  def request(self, input: pd.Series) -> Dict:
    filename = input['filename']
    with open(filename, 'rb') as f:
      response = requests.post(self._url, data=f, headers=self._headers)
    response.raise_for_status()
    return response.json()
