import pandas as pd
import requests

from aidb.inference.http_inference_service import HTTPInferenceService

class GoogleVisionAnnotate(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      project_id: str,
      infer_type: str='images',
  ):
    '''
    :param str token: The token to use for authentication.
    :param str project_id: The project ID to use for authentication.
    '''
    assert infer_type in [
        "images", "files"
      ], "infer_type must be images"
    super().__init__(
        name='google_vision_image_annotate',
        url=f'https://vision.googleapis.com/v1/{infer_type}:annotate',
        headers={
          'Content-Type': 'application/json; charset=utf-8',
          'Authorization': f'Bearer {token}',
          'x-goog-user-project': project_id,
        },
        copy_input=False,
        batch_supported=False,
        is_single=False,
    )
    self._project_id = project_id
    self.infer_type = infer_type


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    input_dict = input.to_dict()
    input_json = {
      'requests': [{
        'features': input_dict['features'],
      },],
      'parent': f'projects/{self._project_id}'
    }
    if 'imageContext' in input_dict:
      input_json['requests'][0]['imageContext'] = input_dict['imageContext']
    match self.infer_type:
      case "files":
        if 'pages' in input_dict:
          input_json['requests'][0]['pages'] = input_dict['pages']
        if 'gcsSource' in input_dict:
          input_json['requests'][0]['inputConfig'] = {
            'gcsSource': {
              'uri': input_dict['gcsSource']
            }
          }
        elif 'content' in input_dict:
          input_json['requests'][0]['inputConfig'] = {
            'content': input_dict['content']
          }
        else:
          raise ValueError('input must contain gcsSource or content')
        if 'mimeType' in input_dict:
          input_json['requests'][0]['inputConfig']['mimeType'] = input_dict['mimeType']
      case "images": 
        if 'image' in input_dict:
          input_json['requests'][0]['image'] = {
            'content': input_dict['image']
          }
        elif 'gcsImageUri' in input_dict:
          input_json['requests'][0]['image'] = {
            'source': {
              'gcsImageUri': input_dict['gcsImageUri']
            }
          }
        elif 'imageUri' in input_dict:
          input_json['requests'][0]['image'] = {
            'source': {
              'imageUri': input_dict['imageUri']
            }
          }
        else:
          raise ValueError('input must contain image, gcsImageUri, or imageUri')
    response = requests.post(self._url, json=input_json, headers=self._headers)
    response.raise_for_status()
    response = response.json()
    return response
