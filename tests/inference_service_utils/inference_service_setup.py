import glob
import os
import pandas as pd

from aidb.config.config_types import InferenceBinding, AIDBListType
from aidb.engine import Engine
from aidb.inference.http_inference_service import HTTPInferenceService


def register_inference_services(engine: Engine, data_dir: str):
  csv_fnames = glob.glob(f'{data_dir}/inference/*.csv')
  csv_fnames.sort()  # TODO: sort by number
  for csv_fname in csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    df = pd.read_csv(csv_fname)
    columns = df.columns
    input_cols = []
    output_cols = []
    columns_to_input_keys = {}
    output_keys_to_columns = {}
    in_col_idx = 0
    out_col_idx = 0
    for col in columns:
      if col.startswith("in__"):
        columns_to_input_keys[in_col_idx] = col[4:]
        input_cols.append(col[4:])
        in_col_idx += 1
      elif col.startswith("out__"):
        output_keys_to_columns[(col[5:], AIDBListType())] = out_col_idx
        output_cols.append(col[5:])
        out_col_idx += 1
      else:
        raise Exception("Invalid column name, column name should start with in__ or out__")

    service = HTTPInferenceService(
      service_name,
      False,
      url=f'http://127.0.0.1:8000/{service_name}',
      headers={'Content-Type': 'application/json'},
      columns_to_input_keys=columns_to_input_keys,
      response_keys_to_columns=output_keys_to_columns
    )

    engine.register_inference_service(service)
    engine.bind_inference_service(
      service_name,
      InferenceBinding(
        tuple(input_cols),
        tuple(output_cols),
      )
    )
