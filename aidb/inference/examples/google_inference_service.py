from typing import Dict, Union

import subprocess

from aidb.inference.examples.https_inference_service import HTTPSInferenceService


def get_gcloud_access_token():
  try:
    command = "gcloud auth application-default print-access-token"
    result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    token = result.stdout.strip() 
    return token

  except subprocess.CalledProcessError as e:
    print(f"An error occurred while getting token: {str(e)}")
    print(f"stderr: {e.stderr.strip()}")
    return None


class GoogleVisionAnnotate(HTTPSInferenceService):
  def __init__(
      self,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      project_id: str=None,
      infer_type: str='images',
  ):
    assert infer_type in [
        "images", "files"
      ], "infer_type must be images or files"
    assert project_id is not None, "project_id must be specified"
    if token is None:
      token = get_gcloud_access_token()
    super().__init__(
        name='google_vision_image_annotate',
        url=f'https://vision.googleapis.com/v1/{infer_type}:annotate',
        headers={
          'Content-Type': 'application/json; charset=utf-8',
          'Authorization': f'Bearer {token}',
          'x-goog-user-project': project_id,
        },
        default_args=default_args,
        copy_input=False,
        batch_supported=False,
        is_single=False,
        columns_to_input_keys=columns_to_input_keys,
        response_keys_to_columns=response_keys_to_columns,
    )
