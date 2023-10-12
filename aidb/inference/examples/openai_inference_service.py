from typing import Dict, Union

from aidb.inference.http_inference_service import HTTPInferenceService


class OpenAIAudio(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      columns_to_input_keys: Dict[str, Union[str, tuple]],
      response_keys_to_columns: Dict[Union[str, tuple], str],
      infer_type: str='transcriptions'):
    '''
    :param str token: The token to use for authentication.
    :param str infer_type: 'transcriptions'|'translations'.
    '''
    assert infer_type in [
        "transcriptions",
        "translations",
      ], "infer_type must be transcriptions or translations"
    super().__init__(
      name='openai_audio',
      url=f'https://api.openai.com/v1/audio/{infer_type}',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class OpenAIImage(HTTPInferenceService):
  def __init__(
      self, 
      token: str, 
      columns_to_input_keys: Dict[str, Union[str, tuple]],
      response_keys_to_columns: Dict[Union[str, tuple], str],
      infer_type: str='generations'
    ):
    '''
    :param str token: The token to use for authentication.
    :param str infer_type: 'generations'|'edits'|'variations'
    '''
    assert infer_type in [
        "generations",
        "edits",
        "variations",
      ], "infer_type must be generations, edits or variations"
    super().__init__(
      name='openai_image',
      url=f'https://api.openai.com/v1/images/{infer_type}',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )


class OpenAIText(HTTPInferenceService):
  def __init__(
      self, 
      token: str,
      columns_to_input_keys: Dict[str, Union[str, tuple]],
      response_keys_to_columns: Dict[Union[str, tuple], str],
    ):
    '''
    :param str token: The token to use for authentication.
    '''
    super().__init__(
      name='openai_text',
      url='https://api.openai.com/v1/chat/completions',
      headers={
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=response_keys_to_columns,
    )
