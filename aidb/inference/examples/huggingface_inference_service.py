from aidb.inference.http_inference_service import HTTPInferenceService


class HFNLP(HTTPInferenceService):
  def __init__(self, token: str, model: str):
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
    )


class HFVisionAudio(HTTPInferenceService):
  def __init__(self, token: str, model: str):
    '''
    :param str token: The token to use for authentication.
    :param str model: The model to use for inference.
    '''
    super().__init__(
      name='huggingface_vision_audio',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=False,
      is_single=False,
    )
