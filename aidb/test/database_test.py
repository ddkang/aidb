from aidb.test.database import Database
import yaml

yaml_file = ''
with open(yaml_file, 'r') as f:
  od_config = yaml.load(f, Loader=yaml.FullLoader)

engine = Database(
    od_config
)
engine._loop.run_until_complete(engine.create_table())

csv_file = ''
table_name = ''
engine._loop.run_until_complete(engine.insert_table(csv_file, table_name))
