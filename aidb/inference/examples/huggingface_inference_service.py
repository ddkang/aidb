import requests
import pandas as pd

from aidb.inference.http_inference_service import HTTPInferenceService


class HFNLP(HTTPInferenceService):
  def __init__(self, token: str, model: str):
    '''
    :param str token: The token to use for authentication.
    :param str model: The model to use for inference.

    All NLP API endpoints accept `inputs`, `parameters` and `options` as JSON-encoded parameters.
    See https://huggingface.co/docs/api-inference/detailed_parameters#natural-language-processing
    '''
    super().__init__(
      name='huggingface_nlp',
      url=f'https://api-inference.huggingface.co/models/{model}',
      headers={
        'Content-Type': 'application/json; charset=utf-8',
        'Authorization': f'Bearer {token}',
      },
      copy_input=False,
      batch_supported=True,
      is_single=False,
    )


class HFVisionAudio(HTTPInferenceService):
  def __init__(self, token: str, model: str):
    '''
    :param str token: The token to use for authentication.
    :param str model: The model to use for inference.

    All Vision/Audio API endpoints only accept `filename` and send binary representation.
    See https://huggingface.co/docs/api-inference/detailed_parameters#audio
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


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    filename = input['filename']
    with open(filename, 'rb') as f:
      response = requests.post(self._url, data=f, headers=self._headers)
    response.raise_for_status()
    response = response.json()
    output = pd.DataFrame([response])
    return output
