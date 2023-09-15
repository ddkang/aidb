import glob
import os
import pandas as pd

from aidb.config.config_types import InferenceBinding
from aidb.engine import Engine
from aidb.inference.http_inference_service import HTTPInferenceService


def register_inference_services(engine: Engine, data_dir: str):
  csv_fnames = glob.glob(f'{data_dir}/inference/*.csv')
  csv_fnames.sort()  # TODO: sort by number
  inference_service_meta_data = pd.read_csv(f'{data_dir}/metadata.csv')
  inference_service_meta_data.set_index("inference_service", drop=True, inplace=True)
  for csv_fname in csv_fnames:
    base_fname = os.path.basename(csv_fname)
    service_name = base_fname.split('.')[0]
    service = HTTPInferenceService(
      service_name,
      False,
      url=f'http://mockurl.com/{service_name}',
      headers={'Content-Type': 'application/json'},
    )

    engine.register_inference_service(service)
    df = pd.read_csv(csv_fname)
    input_columns = df.columns[:inference_service_meta_data.loc[service_name]["num_inputs"]]
    output_columns = df.columns[inference_service_meta_data.loc[service_name]["num_inputs"]:]
    engine.bind_inference_service(
      service_name,
      InferenceBinding(
        tuple(input_columns),
        tuple(output_columns),
      )
    )
