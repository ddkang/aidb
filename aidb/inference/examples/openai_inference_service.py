from typing import Dict, Union

import os

from aidb.inference.http_inference_service import HTTPInferenceService


class OpenAIAudio(HTTPInferenceService):
  def __init__(
      self,
      name: str='openai_audio',
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
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
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class OpenAIImage(HTTPInferenceService):
  def __init__(
      self,
      name='openai_image', 
      token: str=None, 
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
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
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class OpenAIText(HTTPInferenceService):
  def __init__(
      self,
      name: str='openai_text',
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
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
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )
