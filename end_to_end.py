import argparse
import asyncio
import os
from omegaconf import OmegaConf

from aidb.config.config_types import AIDBListType, InferenceBinding
from aidb.engine import Engine
from aidb.inference.examples.huggingface_inference_service import \
  HuggingFaceNLP
from tests.utils import command_line_utility, setup_aidb_engine


async def one_time_aidb_setup(DB_URL, DB_NAME):
  dirname = os.path.dirname(__file__)
  data_dir = os.path.join(dirname, 'data/amazon')
  await setup_aidb_engine(DB_URL, DB_NAME, data_dir)


def main(args):
  in_config  = OmegaConf.load(args.in_config)
  if in_config:
    DB_URL = in_config.db_url
    DB_NAME = in_config.db_name
    input_col = in_config.input_col
    output_col = in_config.output_col
    inference_service_config = in_config.inference_service
  else:
    DB_URL = args.db_url
    DB_NAME = args.db_name
    input_col = args.input_col
    output_col = args.output_col
    inference_service_config = None

  asyncio.run(one_time_aidb_setup(DB_URL, DB_NAME))

  aidb_engine = Engine(
    f'{DB_URL}/{DB_NAME}',
    debug=False,
  )

  if inference_service_config:
    sentiment_classification = HuggingFaceNLP(**inference_service_config)
  else:
    sentiment_classification = HuggingFaceNLP(
      name="sentiment_classification",
      columns_to_input_keys={'blobs00.review': 'inputs'},
      response_keys_to_columns={(AIDBListType(), AIDBListType(), 'label'): 'sentiment.label',
                                (AIDBListType(), AIDBListType(), 'score'): 'sentiment.score'},
      model="LiYuan/amazon-review-sentiment-analysis",
      default_args={"options": {"wait_for_model": True}},
      copy_input=True)

  aidb_engine.register_inference_service(sentiment_classification)
  aidb_engine.bind_inference_service(sentiment_classification.name,
                                     InferenceBinding(input_col, output_col))
  if args.out_config and not in_config:
    out_config = OmegaConf.create({
      "db_url": DB_URL,
      "db_name": DB_NAME,
      "input_col": input_col,
      "output_col": output_col,
      "inference_service": sentiment_classification.to_dict()
    })
    OmegaConf.save(out_config, args.out_config)

  command_line_utility(aidb_engine)



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--in_config", type=str)
  parser.add_argument("--out_config", type=str)    
    
  parser.add_argument("--db_url", type=str)
  parser.add_argument("--db_name", type=str)
  parser.add_argument("--input_col", type=tuple)
  parser.add_argument("--output_col", type=tuple)

  args = parser.parse_args()
  main(args)
