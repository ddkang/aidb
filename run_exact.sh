export DATASET="twitter" # dataset = ['jackson_all', 'law', 'arxiv', 'twitter']
export PORT=8000
export AIDB_NUMBER_OF_TEST_RUNS=100
export BUDGET=5000 # The budget for generating proxy scores using the TASTI algorithm
python3.9 -m tests.vldb_tests.test_approx_select