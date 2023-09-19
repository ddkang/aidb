import glob
import os
import pandas as pd
import uvicorn

from fastapi import FastAPI, Request

from multiprocessing import Process

def run_server(data_dir: str):
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

  # config = Config(app=app, host="127.0.0.1", port=8000)
  # server = Server(config=config)
  uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__=='__main__':
  p = Process(target=run_server, args=["/home/akash/Documents/aidb-new/tests/data/jackson"])
  p.start()
  print("absbdsadhgfahjdsgfahjsdgfhjagsdhjfagdsjhfgahjkdfgajhksdgfakjhsdfgaksjdg")
  p.terminate()