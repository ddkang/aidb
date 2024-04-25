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
    if service_name.endswith('__join'):
      service_name = service_name[:-6]
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
        group = group.drop(columns=name_to_input_cols[service_name]).dropna(how='all')
        res_df_list.append(group.to_dict(orient='list'))

      return res_df_list


    # inference service for JOIN query
    @app.post(f'/{service_name}__join')
    async def join_inference(inp: Request):
      service_name = inp.url.path.split('/')[-1]
      service_name = service_name[:-6]

      inp = await inp.json()
      df = name_to_df[service_name]

      # Construct a DataFrame from the input
      inp_df = pd.DataFrame({col: [inp[col]] if not isinstance(inp[col], list) else inp[col]
                             for col in name_to_input_cols[service_name]})

      # Performing the merge
      # Note: We're using a left join to ensure that all inputs have corresponding outputs,
      #     with absent outputs represented as None
      # For JOIN query, each input pair has exact one output
      result_df = pd.merge(inp_df, df, how='left', on=name_to_input_cols[service_name]).convert_dtypes()

      # For the JOIN service, the table stores only True values. We need to populate it with False values for the remaining inputs.
      inference_cols = []
      service_output_cols = []
      output_col_rename = []
      output_to_input_mapping = {col.split('.')[1]: col for col in name_to_input_cols[service_name]}
      for col in df.columns:
        if col not in inp_df.columns:
          if col.split('.')[1] in output_to_input_mapping:
            service_output_cols.append(output_to_input_mapping[col.split('.')[1]])
          else:
            inference_cols.append(col)
            service_output_cols.append(col)
          output_col_rename.append(col)

      # Copy the keys from the inputs, as they will default to None if the inference result is False.
      result_df = result_df[service_output_cols]
      result_df.columns = output_col_rename
      for col in inference_cols:
        if pd.api.types.is_bool_dtype(result_df[col]):
          result_df[col] = result_df[col].fillna(False)
        else:
          result_df[col] = result_df[col].fillna(0).astype(bool)
      return result_df.to_dict(orient='list')


  # config = Config(app=app, host="127.0.0.1", port=8000)
  # server = Server(config=config)
  uvicorn.run(app, host="127.0.0.1", port=port, log_level='warning')

if __name__=='__main__':
  run_server(sys.argv[1])
