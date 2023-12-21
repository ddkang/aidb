from typing import Dict, Union, List

import subprocess

from aidb.inference.http_inference_service import HTTPInferenceService


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


class GoogleVisionAnnotate(HTTPInferenceService):
  def __init__(
      self,
      name: str='google_vision_annotate',
      is_single: bool=False,
      token: str=None,
      default_args: Dict[str, Union[str, int]]=None,
      columns_to_input_keys: Dict[str, Union[str, tuple]]=None,
      response_keys_to_columns: Dict[Union[str, tuple], str]=None,
      input_columns_types: Union[List, None]=None,
      output_columns_types: Union[List, None]=None,
      copied_input_columns: List[int]=[],
      rate_limit: Union[int, None]=None,
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
        name=name,
        url=f'https://vision.googleapis.com/v1/{infer_type}:annotate',
        headers={
          'Content-Type': 'application/json; charset=utf-8',
          'Authorization': f'Bearer {token}',
          'x-goog-user-project': project_id,
        },
        default_args=default_args,
        batch_supported=False,
        is_single=is_single,
        rate_limit=rate_limit,
        columns_to_input_keys=columns_to_input_keys,
        response_keys_to_columns=response_keys_to_columns,
        input_columns_types=input_columns_types,
        output_columns_types=output_columns_types,
        copied_input_columns=copied_input_columns,
    )
