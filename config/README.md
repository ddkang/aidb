# Configure and Run AIDB

## Run AIDB with Configuration

Configuration file for AIDB is a Python module imported at runtime. Specify the module of your configuration via `--config`. Running AIDB with
```bash
python3 launch.py --config=config.sentiment
```
means running AIDB with configuration located at `config/sentiment.py`.

You may optionally set up blob table and output tables with `--setup-blob-table` and `--setup-output-tables`, respectively. See [Table Configuration](#table-configuration) for details. For example, running AIDB with
```bash
python3 launch.py --config=config.sentiment --setup-blob-table --setup-output-tables
```
means running AIDB with configuration located at `config/sentiment.py` and setting up blob table and output tables according to your configuration.

You may optionally add `--verbose` to your command line arguments to see the progress of your query execution.

## Inference Service Configuration

In this section, we will walk through an example of configuring an inference service for AIDB.

1. Import the inference service you want to use. Here we import `HuggingFaceNLP` from `aidb/inference/examples/huggingface_inference_service.py`.
    ```python
    from aidb.inference.examples.huggingface_inference_service import HuggingFaceNLP
    ```

2. Define inference service. Here we define a sentiment analysis service from `HuggingFace`. Read [here](https://github.com/ddkang/aidb-new/tree/main/aidb/inference/examples) for how to fill in attributes for each inference service and how to define key map between JSON and pandas DataFrame.
    ```python
    from aidb.config.config_types import AIDBListType # utility class for key map conversion
    sentiment_inference_service = HuggingFaceNLP( # you can change variable name and class name
      name="sentiment_classification", # define your own name here
      token=HF_TOKEN, # fill in your own token here
      columns_to_input_keys=['inputs'], # input -> json key map
      response_keys_to_columns=[ # json -> output key map
        (AIDBListType(), AIDBListType(), 'label'),
        (AIDBListType(), AIDBListType(), 'score')],
      input_columns_types=[str], # optional, input type check
      output_columns_types=[str, float], # optional, output type check
      model="LiYuan/amazon-review-sentiment-analysis", # model for hf
      default_args={("options", "wait_for_model"): True} # default args
    )
    ```

3. Define inference engines. It is a list of dictionary. Each dictionary contains a service (`service`), a binding for input columns (`input_col`) and a binding for output columns (`output_col`).
    ```python
    # keep variable name as "inference_engines"
    inference_engines = [
      {
        "service": sentiment_inference_service, # keep key name as "service", change value according to your service defined above
        "input_col": ("blobs00.review", "blobs00.review_id"), # keep key name as "input_col", change value according to your input binding
        "output_col": ("sentiment.label", "sentiment.score", "sentiment.review_id") # keep key name as "output_col", change value according to your output binding
      } # you can add multiple inference engines after the above one, following the format above.
    ]
    ```
    Input and output bindings are tuples of string, with each binding named according to `table.column`. For example, `blobs00.review` is the `review` column from table `blobs00`.
    
    Order matters when you define bindings. In the above example,
    - `blobs00.review` and `blobs00.review_id` will be passed to the inference service as the first and second argument, respectively. Only `blobs00.review` will be converted to JSON request according to `columns_to_input_keys` map. 
    - The output of the inference service will be passed to `sentiment.label`, `sentiment.score` and `sentiment.review_id` as the first, second and third argument, respectively. `sentiment.label` and `sentiment.score` are from the inference service - they are converted from JSON to pandas DataFrame according to `response_keys_to_columns` map. `sentiment.review_id` is copied from `blobs00.review_id`. Keep column name the same if you want to copy from input.

4. Set up your database configuration, including the URL to your database and its name.
    ```python
    DB_URL = 'sqlite+aiosqlite://' # database url, keep variable name as "DB_URL", change value
    DB_NAME = 'aidb_test.sqlite' # database name, keep variable name as "DB_NAME", change value
    ```

## Table Configuration

Read this section if you want to set up tables according to a csv file. We will walk through an example of configuring table set-up for AIDB.

- Set up blob table. Run AIDB with command line argument `--setup-blob-table` once. Future runs with this argument will not set up blob table again unless you remove your database file.
    ```python
    blobs_csv_file = "tests/data/amazon_reviews.csv" # path to csv file, keep variable name as "blobs_csv_file", change value
    blob_table_name = "blobs00" # table name, keep variable name as "blob_table_name", change value
    blobs_keys_columns = ["review_id"] # primary keys, keep variable name as "blobs_keys_columns", change value
    ```

- Set up output tables. Run AIDB with command line argument `--setup-output-tables` once. Future runs with this argument will not set up output tables again unless you remove your database file.
    ```python
    # keep variable name as "tables", change value according to your output tables
    tables = {"sentiment": [ # table name
      {
        "name": "review_id", # column name, keey key name as "name", change value
        "is_primary_key": True, # default False, keep key name as "is_primary_key", change value
        "refers_to": ("blobs00", "review_id"), # foreign key in format (referenced_table_name, referenced_column_name), default None (no reference), keep key name as "refers_to", change value
        "dtype": int # data type of the column, keep key name as "dtype", change value
      },
      {
        "name": "label",
        "is_primary_key": True,
        "dtype": str,
      },
      {
        "name": "score",
        "dtype": float
      }
    ]}
    ```
