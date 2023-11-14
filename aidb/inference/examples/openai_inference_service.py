from typing import Dict, Union, List

import os

import pandas as pd

from aidb.inference.http_inference_service import HTTPInferenceService


class OpenAIAudio(HTTPInferenceService):
  def __init__(
      self,
      name: str='openai_audio',
      is_single: bool=False,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      gap_between_requests: float=0.0,
      input_columns_types: Union[List, None]=None,
      output_columns_types: Union[List, None]=None,
      infer_type: str='transcriptions'):
    assert infer_type in [
        "transcriptions",
        "translations",
      ], "infer_type must be transcriptions or translations"
    if token is None:
      token = os.environ['OPENAI_API_KEY']
    super().__init__(
      name=name,
      url=f'https://api.openai.com/v1/audio/{infer_type}',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      batch_supported=False,
      is_single=is_single,
      gap_between_requests=gap_between_requests,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
      input_columns_types=input_columns_types,
      output_columns_types=output_columns_types,
    )


class OpenAIImage(HTTPInferenceService):
  def __init__(
      self,
      name='openai_image', 
      is_single: bool=False,
      token: str=None, 
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      gap_between_requests: float=0.0,
      input_columns_types: Union[List, None]=None,
      output_columns_types: Union[List, None]=None,
      infer_type: str='generations'
    ):
    assert infer_type in [
        "generations",
        "edits",
        "variations",
      ], "infer_type must be generations, edits or variations"
    if token is None:
      token = os.environ['OPENAI_API_KEY']
    super().__init__(
      name=name,
      url=f'https://api.openai.com/v1/images/{infer_type}',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      batch_supported=False,
      is_single=is_single,
      gap_between_requests=gap_between_requests,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
      input_columns_types=input_columns_types,
      output_columns_types=output_columns_types,
    )


class OpenAIText(HTTPInferenceService):
  def __init__(
      self,
      name: str='openai_text',
      is_single: bool=False,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      gap_between_requests: float=0.0,
      input_columns_types: Union[List, None]=None,
      output_columns_types: Union[List, None]=None,
      prompt_prefix: str='',
      prompt_suffix: str='',
    ):
    if token is None:
      token = os.environ['OPENAI_API_KEY']
    super().__init__(
      name=name,
      url='https://api.openai.com/v1/chat/completions',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      default_args=default_args,
      batch_supported=False,
      is_single=is_single,
      gap_between_requests=gap_between_requests,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
      input_columns_types=input_columns_types,
      output_columns_types=output_columns_types,
    )
    self.prompt_prefix = prompt_prefix
    self.prompt_suffix = prompt_suffix


  def convert_input_to_request(self, input: Union[pd.Series, pd.DataFrame]) -> Dict:
    request = super().convert_input_to_request(input)
    if 'messages' in request and isinstance(request['messages'], list):
      for i in range(len(request['messages'])):
        if 'content' in request['messages'][i]:
          request['messages'][i]['content'] = self.prompt_prefix + request['messages'][i]['content'] + self.prompt_suffix
    return request
