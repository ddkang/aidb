<h1 style="text-align: center;">AIDB</h1>

<p align="center"> Analyze unstructured data blazingly fast with machine learning. Connect your own ML models to your own data sources and query away! </p>

<p align="center">
  <img src="assets/aidbuse.gif" style="width:550px;"/>
</p>

## Quick Start

In order to start using AIDB, all you need to do is install the requirements, specify a configuration, and query!
Setting up on the environment is as simple as
```bash
git clone https://github.com/ddkang/aidb.git
cd aidb
pip install -r requirements.txt

# Optional if you'd like to run the examples below
gdown https://drive.google.com/uc?id=1SyHRaJNvVa7V08mw-4_Vqj7tCynRRA3x
unzip data.zip -d tests/

```

### Text Example (in CSV)

We've set up an example of analyzing product reviews with HuggingFace. Set your HuggingFace API key. After this, all you need to do is run
```bash
python launch.py --config=config.sentiment --setup-blob-table --setup-output-table
```

As an example query, you can run
```sql
SELECT AVG(score)
FROM sentiment
WHERE label = '5 stars'
ERROR_TARGET 10%
CONFIDENCE 95%;
```

You can see the mappings [here](https://github.com/ddkang/aidb/blob/main/config/sentiment.py#L15). We use the HuggingFace API to generate sentiments from the reviews.


### Image Example (local directory)

We've also set up another example of analyzing whether or not user-generated content is adult content for filtering.
In order to run this example, all you need to do is run
```bash
python launch.py --config=config.nsfw_detect --setup-blob-table --setup-output-table
```

As an example query, you can run
```sql
SELECT *
FROM nsfw
WHERE racy LIKE 'POSSIBLE';
```

You can see the mappings [here](https://github.com/ddkang/aidb/blob/main/config/nsfw_detect.py#L10). We use the Google Vision API to generate the safety labels.



## Key Features

AIDB focuses on keeping cost down and interoperability high.

We reduce costs with our optimizations:
- First-class support for approximate queries, reducing the cost of aggregations by up to **350x**.
- Caching, which speeds up multiple queries over the same data.

We keep interoperability high by allowing you to bring your own data source, ML models, and vector databases!


## Approximate Querying

One key feature of AIDB is first-class support for approximate queries.
Currently, we support approximate `AVG`, `COUNT`, and `SUM`.
We don't currently support `GROUP BY` or `JOIN` for approximate aggregations, but it's on our roadmap.
Please reach out if you'd like us to support your queries!

In order to execute an approximate aggregation query, simply append `ERROR_TARGET <error percent>% CONFIDENCE <confidence>%` to your normal aggregation.
As a full example, you can compute an approximate count by doing:
```sql
SELECT COUNT(xmin)
FROM objects
ERROR_TARGET 5%
CONFIDENCE 95%;
```

The `ERROR_TARGET` specifies the percent error _compared to running the query exactly._
For example, if the true answer is 100, you will get answers between 95 and 105 (95% of the time).


## Useful Links
- [How to connect ML APIs](https://github.com/ddkang/aidb/blob/main/aidb/inference/examples/README.md) 
- [How to connect to Datastore]() - TODO
- [How to define configuration file](https://github.com/ddkang/aidb/tree/main/config)

## Contribute

We have many improvements we'd like to implement. Please help us! For the time being, please [email](mailto:ddkang@g.illinois.edu) us, if you'd like to help contribute.


## Contact Us

Need help in setting up AIDB for your specific dataset or want a new feature? Please fill [this form](https://forms.gle/YyAXWxqzZPVBrvBR7).
