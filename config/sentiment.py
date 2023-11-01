from aidb.config.config_types import AIDBListType
from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP


DB_URL = 'sqlite+aiosqlite://'
DB_NAME = 'aidb_test_amazon.sqlite'
DATA_DIR = 'tests/data/amazon'
HF_KEY = '<hf key>'


sentiment_inference_service = HuggingFaceNLP(
      name="sentiment_classification",
      token=HF_KEY,
      columns_to_input_keys=['inputs'],                                     # list index 0 means the first column of the input dataframe, but the actual first column has name "blobs00.review"
      response_keys_to_columns=[(AIDBListType(), AIDBListType(), 'label'),  # list index 0 means the first column of the output dataframe, but the actual first column has name "sentiment.label" 
                                (AIDBListType(), AIDBListType(), 'score')], # list index 1 means the second column of the output dataframe, but the actual second column has name "sentiment.score"
      input_columns_types=[str],
      output_columns_types=[str, float],
      model="LiYuan/amazon-review-sentiment-analysis",
      default_args={"options": {"wait_for_model": True}})
inference_engines = [
  {
    "service": sentiment_inference_service,
    "input_col": ("blobs00.review", "blobs00.review_id"), # I leave the bindings as they are for now, but they may change following the change above
    "output_col": ("sentiment.label", "sentiment.score", "sentiment.review_id")
  }
]