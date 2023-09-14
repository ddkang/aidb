import glob
import os

import pandas as pd
from fastapi import FastAPI, Request


# parser = argparse.ArgumentParser(description='FastAPI App')
# parser.add_argument('--data_dir', required=True)
# args = parser.parse_args()
# 
# data_dir = args.data_dir
# TODO: uvicorn doesn't work with argparse
# data_dir = '/home/ubuntu/aidb/aidb-new/tests/data/jackson'
data_dir = "/home/akash/Documents/aidb-new/data/jackson"
app = FastAPI()

inference_dir = f'{data_dir}/inference'
inference_csv_fnames = glob.glob(f'{inference_dir}/*.csv')
inference_csv_fnames.sort()

# Create the inference services
name_to_df = {}
for csv_fname in inference_csv_fnames:
  base_fname = os.path.basename(csv_fname)
  service_name = base_fname.split('.')[0]
  df = pd.read_csv(csv_fname)
  name_to_df[service_name] = df
  print('Creating service', service_name)
  @app.post(f'/{service_name}')
  async def inference(inp: Request):
    service_name = inp.url.path.split('/')[-1]
    try:
      inp = await inp.json()
      # Get the rows where the input columns match
      df = name_to_df[service_name]
      tmp = df[df[df.columns[0]] == inp[0]]
      for column, value in zip(df.columns[1:], inp[1:]):
        tmp = tmp[tmp[column] == value]
      res = tmp.to_dict(orient='list')
      print(res)
      return res
    except Exception as e:
      print('Error', e)
      return []
