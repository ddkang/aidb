import pandas as pd
from aidb.inference.http_inference_service import HTTPInferenceService


class OpenAIAudio(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      infer_type: str='transcriptions'
  ):
    '''
    :param str token: The token to use for authentication.
    :param str inference_type: The type of inference to perform. Either 'transcriptions' or 'translations'.
    '''
    assert infer_type in [
        "transcriptions",
        "translations",
      ], "infer_type must be either transcriptions or translations"
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
      columns_to_input_keys={
        'file': 'file',
        'model': 'model',
        'prompt': 'prompt',
        'response_format': 'response_format',
        'temperature': 'temperature',
        'language': 'language',
      },
      response_keys_to_columns={
        'text': 'text',
      },
    )
    self.infer_type = infer_type


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    assert 'file' in input, 'file is a required column'
    assert 'model' in input, 'model is a required column'
    assert input['model'] == 'whisper-1', 'model must be whisper-1'
    assert self.infer_type == 'transcriptions' or 'language' not in input, 'language is only supported for translations'
    super().infer_one(input)
