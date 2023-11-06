from aidb.config.config_types import AIDBListType
from aidb.inference.examples.google_inference_service import GoogleVisionAnnotate

DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb_test_nsfw.sqlite'

nsfw_detect_service = GoogleVisionAnnotate(
  name="nsfw_detect",
  token=None, # automatically get token from gcloud
  columns_to_input_keys=[
    ('requests', AIDBListType(), 'image', 'source', 'imageUri')],
  response_keys_to_columns=[
    ('responses', AIDBListType(), 'safeSearchAnnotation', 'adult'),
    ('responses', AIDBListType(), 'safeSearchAnnotation', 'spoof'),
    ('responses', AIDBListType(), 'safeSearchAnnotation', 'medical'),
    ('responses', AIDBListType(), 'safeSearchAnnotation', 'violence'),
    ('responses', AIDBListType(), 'safeSearchAnnotation', 'racy')],
  input_columns_types=[str],
  output_columns_types=[str, str, str, str, str],
  project_id="coral-sanctuary-400802",
  default_args={('requests', AIDBListType(), 'features', 'type'): 'SAFE_SEARCH_DETECTION',
                'parent': 'projects/coral-sanctuary-400802'})
inference_engines = [
  {
    "service": nsfw_detect_service,
    "input_col": ("blobs00.image_url", "blobs00.image_id"),
    "output_col": ("nsfw.adult", "nsfw.spoof", "nsfw.medical", "nsfw.violence", "nsfw.racy", "nsfw.image_id")
  }
]

blobs_csv_file = "image_path_data.csv"
blob_table_name = "image"
blobs_keys_columns = ["image_id"]

"""
dictionary of table names to list of columns
"""

tables = {"nsfw": [
  {"name": "image_id", "is_primary_key": True, "refers_to": ("image", "image_id"), "dtype": int},
  {"name": "adult", "dtype": str},
  {"name": "spoof", "dtype": str},
  {"name": "medical", "dtype": str},
  {"name": "violence", "dtype": str},
  {"name": "racy", "dtype": str}]}
