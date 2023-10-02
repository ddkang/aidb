import pandas as pd
import requests

from aidb.inference.http_inference_service import HTTPInferenceService

class GoogleVisionImageAnnotate(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      project_id: str,
  ):
    '''
    :param str token: The token to use for authentication.
    :param str project_id: The project ID to use for authentication.
    '''
    super().__init__(
        name='google_vision_image_annotate',
        url='https://vision.googleapis.com/v1/images:annotate',
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


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    input_dict = input.to_dict()
    input_json = {
      'requests': [
        {
          'image': {
            'content': input_dict['image'],
          },
          'features': [
            {
              'type': input_dict['feature_type'],
            },
          ],
        },
      ],
      'parent': f'projects/{self._project_id}'
    }
    response = requests.post(self._url, json=input_json, headers=self._headers)
    response.raise_for_status()
    response = response.json()
    response = response['responses'][0]['faceAnnotations'][0]['boundingPoly']['vertices']
    face_annotation = {
      'x_min': response[0]['x'],
      'y_min': response[0]['y'],
      'x_max': response[2]['x'],
      'y_max': response[2]['y'],
    }
    return pd.DataFrame([face_annotation])
