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
  preferred_batch_size=100,
  copied_input_columns=[1],
  project_id="your-project-id",
  default_args={('requests', AIDBListType(), 'features', 'type'): 'SAFE_SEARCH_DETECTION',
                'parent': 'projects/your-project-id'})
inference_engines = [
  {
    "service": nsfw_detect_service,
    "input_col": ("images_source.image_url", "images_source.image_id"),
    "output_col": ("images.adult", "images.spoof", "images.medical", "images.violence", "images.racy", "images.image_id")
  }
]

blobs_csv_file = "tests/data/image_path_data.csv"
blob_table_name = "images_source"
blobs_keys_columns = ["image_id"]

"""
dictionary of table names to list of columns
"""

tables = {"images": [
  {"name": "image_id", "is_primary_key": True, "refers_to": ("images_source", "image_id"), "dtype": int},
  {"name": "adult", "dtype": str},
  {"name": "spoof", "dtype": str},
  {"name": "medical", "dtype": str},
  {"name": "violence", "dtype": str},
  {"name": "racy", "dtype": str}]}
