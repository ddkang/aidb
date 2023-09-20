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
  name_to_input_cols = {}
  name_to_output_cols = {}
  for csv_fname in inference_csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    df = pd.read_csv(csv_fname)
    new_col_names = []
    output_cols = []
    input_cols = []
    for col in df.columns:
      if col.startswith("in__"):
        new_col_names.append(col[4:])
        input_cols.append(col[4:])
      elif col.startswith("out__"):
        new_col_names.append(col[5:])
        output_cols.append(col[5:])
      else:
        raise Exception("Column doesn't start with in__ or out__")
    df.columns = new_col_names
    name_to_df[service_name] = df
    name_to_input_cols[service_name] = input_cols
    name_to_output_cols[service_name] = output_cols
    print('Creating service', service_name)

    @app.post(f'/{service_name}')
    async def inference(inp: Request):
      service_name = inp.url.path.split('/')[-1]
      try:
        inp = await inp.json()
        # Get the rows where the input columns match
        df = name_to_df[service_name]
        tmp = df
        for col in name_to_input_cols[service_name]:
          tmp = tmp[tmp[col] == inp[col]]
        res = tmp[name_to_output_cols[service_name]].to_dict(orient='list')
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