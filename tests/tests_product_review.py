import asyncio

from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
from aidb.engine import Engine
from aidb.config.config_types import InferenceBinding
import os

from tests.utils import setup_aidb_engine, command_line_utility

hf_key = "<Your API Key>"

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
    columns_to_input_keys={'review': 'inputs'},
    response_keys_to_columns={('0', '0', 'label',): 'label', ('0', '0', 'score'): 'score'},
    model="LiYuan/amazon-review-sentiment-analysis")

  aidb_engine.register_inference_service(sentiment_classification)
  aidb_engine.bind_inference_service("sentiment_classification",
                                     InferenceBinding(("blobs00.review_id", "blobs00.review"),
                                                      ("sentiment.review_id", "sentiment.label", "sentiment.score")))
  command_line_utility(aidb_engine)
