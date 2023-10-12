from typing import Dict, Union

from aidb.inference.http_inference_service import HTTPInferenceService


class HuggingFaceService(HTTPInferenceService):
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
      name='huggingface_service',
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
