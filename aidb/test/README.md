# AIDB SetUp

setup database for test via `config.yaml` and `table_name.csv`

## How to Use 

1. put your config.yaml and table_name.csv under your `data_dir`
1. configurate `data_dir` in `database_test.py`
1. configurate the creation of your blob table
1. run `python3 -m aidb.test.database_test`

Note that if a table you want to create already exists in your database, it will be automaticaly dropped.
