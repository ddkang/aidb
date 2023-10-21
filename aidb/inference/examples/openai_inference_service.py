from typing import Dict, Union

import os

from aidb.inference.examples.https_inference_service import HTTPSInferenceService


class OpenAIAudio(HTTPSInferenceService):
  def __init__(
      self,
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
      name='openai_audio',
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


class OpenAIImage(HTTPSInferenceService):
  def __init__(
      self, 
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
      name='openai_image',
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


class OpenAIText(HTTPSInferenceService):
  def __init__(
      self, 
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
    ):
    if token is None:
      token = os.environ['OPENAI_API_KEY']
    super().__init__(
      name='openai_text',
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
