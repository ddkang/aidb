export DATASET="law" # dataset = ['jackson_all', 'law', 'arxiv', 'twitter']
export PORT=8000
export TASK='main' # task = ['main', 'selectivity', 'error']
export AIDB_NUMBER_OF_TEST_RUNS=100
python3.9 -m tests.vldb_tests.test_aggregation