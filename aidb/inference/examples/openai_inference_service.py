import pandas as pd
import requests
from aidb.inference.http_inference_service import HTTPInferenceService


class OpenAIAudio(HTTPInferenceService):
  def __init__(self, token: str, infer_type: str='transcriptions'):
    '''
    :param str token: The token to use for authentication.
    :param str infer_type: 'transcriptions'|'translations'.

    For `infer_one`, input should have 1 row, with keys following format 
    at https://platform.openai.com/docs/api-reference/audio/createTranscription
    or https://platform.openai.com/docs/api-reference/audio/createTranslation

    output has 1 row, and 1 column, with key
    - `text`: returned text transcription or translation.
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
    )


class OpenAIImage(HTTPInferenceService):
  def __init__(self, token: str, infer_type: str='generations'):
    '''
    :param str token: The token to use for authentication.
    :param str infer_type: 'generations'|'edits'|'variations'

    For `infer_one`, input should have 1 row, with keys following format
    at https://platform.openai.com/docs/api-reference/images/create
    or https://platform.openai.com/docs/api-reference/images/createEdit
    or https://platform.openai.com/docs/api-reference/images/createVariation

    output has 1 row, and 2 columns, with keys
    - `created`: timestamp of when the image was created.
    - `url`/`b64_json`: url or base64-encoded image output, depending on input `response_format`.
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
    )


class OpenAIText(HTTPInferenceService):
  def __init__(self, token: str):
    '''
    :param str token: The token to use for authentication.

    For `infer_one`, input can have multiple rows.
      Except `messages` column, all other columns should be the same.
      Format should follow https://platform.openai.com/docs/api-reference/chat/create

    outputs may have multiple rows. Columns containing all `choices` from generated text,
      following https://platform.openai.com/docs/api-reference/chat/object
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
    )


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    request = input.to_dict(orient='records')[0]
    request['messages'] = []
    for message in input['messages']:
      request['messages'].append(message)
    response = requests.post(self._url, json=request, headers=self._headers)
    response.raise_for_status()
    response = response.json()
    output = pd.DataFrame(response['choices'])
    if self._copy_input:
      output = output.assign(**input)
    return output
