import asyncio

import pandas as pd

from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
from aidb.engine import Engine
from aidb.config.config_types import InferenceBinding, AIDBListType
import os

from tests.utils import setup_aidb_engine, command_line_utility

hf_key = "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"

DB_URL = "sqlite+aiosqlite://"
DB_NAME = 'aidb_test_amazon.sqlite'


async def one_time_aidb_setup():
  dirname = os.path.dirname(__file__)
  data_dir = os.path.join(dirname, 'data/amazon')
  await setup_aidb_engine(DB_URL, DB_NAME, data_dir)


if __name__ == '__main__':
  asyncio.run(one_time_aidb_setup())

  aidb_engine = Engine(
    f'{DB_URL}/{DB_NAME}',
    debug=False,
  )

  sentiment_classification = HuggingFaceNLP(
    name="sentiment_classification",
    token=hf_key,
    columns_to_input_keys={'blobs00.review': 'inputs'},
    response_keys_to_columns={(AIDBListType(), AIDBListType(), 'label'): 'sentiment.label',
                              (AIDBListType(), AIDBListType(), 'score'): 'sentiment.score'},
    model="LiYuan/amazon-review-sentiment-analysis",
    default_args={"options": {"wait_for_model": True}},
    copy_input=True)

  # result = sentiment_classification.infer_one(pd.Series({"review": "this looks good", "review_id": 1}))

  aidb_engine.register_inference_service(sentiment_classification)
  aidb_engine.bind_inference_service("sentiment_classification",
                                     InferenceBinding(("blobs00.review_id", "blobs00.review"),
                                                      ("sentiment.review_id", "sentiment.label", "sentiment.score")))
  command_line_utility(aidb_engine)
