import glob
import os
import pandas as pd

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine
from aidb.inference.http_inference_service import HTTPInferenceService


def register_inference_services(engine: Engine, data_dir: str):
  csv_fnames = glob.glob(f'{data_dir}/inference/*.csv')
  csv_fnames.sort()  # TODO: sort by number
  for csv_fname in csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    # FIXME: the url should be updated with the correct url
    service = HTTPInferenceService(
      service_name,
      False,
      url=f'http://mockurl.com/{service_name}',
      headers={'Content-Type': 'application/json'},
    )

    engine.register_inference_service(service)
    df = pd.read_csv(csv_fname)
    columns = df.columns
    input_columns = []
    output_columns = []
    for col in columns:
      if col.startswith("in__"):
        input_columns.append(col[4:])
      elif col.startswith("out__"):
        output_columns.append(col[5:])
      else:
        raise Exception("Invalid column name, column name should start with in__ or out__")
    engine.bind_inference_service(
      service_name,
      InferenceBinding(
        tuple(input_columns),
        tuple(output_columns),
      )
    )
