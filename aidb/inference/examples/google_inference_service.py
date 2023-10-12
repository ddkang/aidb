from typing import Dict, Union

from aidb.inference.http_inference_service import HTTPInferenceService


class GoogleVisionAnnotate(HTTPInferenceService):
  def __init__(
      self,
      token: str,
      columns_to_input_keys: Dict[str, Union[str, tuple]],
      response_keys_to_columns: Dict[Union[str, tuple], str],
      project_id: str,
      infer_type: str='images',
  ):
    '''
    :param str token: The token to use for authentication.
    :param str project_id: The project ID to use for authentication.
    '''
    assert infer_type in [
        "images", "files"
      ], "infer_type must be images or files"
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
        columns_to_input_keys=columns_to_input_keys,
        response_keys_to_columns=response_keys_to_columns,
    )
    self._project_id = project_id
