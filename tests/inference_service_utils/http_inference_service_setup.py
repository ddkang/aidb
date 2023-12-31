import sys
import glob
import os
import pandas as pd
import uvicorn

from fastapi import FastAPI, Request

from multiprocessing import Process
from aidb.utils.logger import logger

def run_server(data_dir: str, port=8000):
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
    logger.info(f'Creating service {service_name}')


    @app.post(f'/{service_name}')
    async def inference(inp: Request):
      service_name = inp.url.path.split('/')[-1]
      inp = await inp.json()
      df = name_to_df[service_name]

      # Construct a DataFrame from the input
      inp_df = pd.DataFrame({col: [inp[col]] if not isinstance(inp[col], list) else inp[col] 
                            for col in name_to_input_cols[service_name]})

      # Performing the merge
      # Note: We're using a left join to ensure that all inputs have corresponding outputs,
      #     with absent outputs represented as None
      result_df = pd.merge(inp_df, df, how='left', on=name_to_input_cols[service_name]).convert_dtypes()
      # The outputs are grouped by input dataframe's primary key
      grouped = result_df.groupby(name_to_input_cols[service_name])
      res_df_list = []
      for _, group in grouped:
        group = group.drop(columns=name_to_input_cols[service_name]).dropna()
        res_df_list.append(group.to_dict(orient='list'))

      return res_df_list


  # config = Config(app=app, host="127.0.0.1", port=8000)
  # server = Server(config=config)
  uvicorn.run(app, host="127.0.0.1", port=port, log_level='warning')

if __name__=='__main__':
  run_server(sys.argv[1])
