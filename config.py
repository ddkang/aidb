from aidb.config.config_types import AIDBListType
from aidb.inference.examples.huggingface_inference_service import \
  HuggingFaceNLP

DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb'
DATA_DIR = 'tests/data/amazon'
input_col = ("blobs00.review_id", "blobs00.review")
output_col = ("sentiment.review_id", "sentiment.label", "sentiment.score")
inference_engine = HuggingFaceNLP(
  name="sentiment_classification",
  columns_to_input_keys={'blobs00.review': 'inputs'},
  response_keys_to_columns={(AIDBListType(), AIDBListType(), 'label'): 'sentiment.label',
                            (AIDBListType(), AIDBListType(), 'score'): 'sentiment.score'},
  model="LiYuan/amazon-review-sentiment-analysis",
  default_args={"options": {"wait_for_model": True}},
  copy_input=True)