import os
import yaml
import numpy as np
import pandas as pd

from aidb.test.database import Database

data_dir = 'data/jackson'
with open(f'{data_dir}/config.yaml', 'r') as f:
  od_config = yaml.load(f, Loader=yaml.FullLoader)

engine = Database(od_config)

# create blob table pandas dataframe
nb_records = 973136
blob_ids = np.linspace(0, nb_records, num=nb_records, dtype=np.int64)
blob_df = pd.DataFrame({ 'frame': blob_ids })
engine._loop.run_until_complete(engine.insert_value(blob_df, 'blob_ids'))

# iterate all csv files under data_dir, assume each csv file has filename f'{table_name}.csv'
for csv_file in os.listdir(data_dir):
  if csv_file.endswith('.csv'):
    table_name = csv_file.split('.')[0]
    if table_name in engine._config['tables']:
      df = pd.read_csv(f'{data_dir}/{csv_file}')
      engine._loop.run_until_complete(engine.insert_value(df, table_name))
