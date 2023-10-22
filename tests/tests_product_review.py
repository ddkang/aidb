import pandas as pd
from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP

hf_key = "<Your API KEY>"

hf_nlp = HuggingFaceNLP(
  token=hf_key,
  columns_to_input_keys={'inputs': 'inputs'},
  response_keys_to_columns={('0', '0', 'label', ): 'label', ('0', '0', 'score'): 'score'},
  model="LiYuan/amazon-review-sentiment-analysis")

hf_nlp_response_pd = hf_nlp.infer_one(pd.Series({"inputs": "this product is good"}))
print(hf_nlp_response_pd)
