# AIDB

Analyze unstructured data blazingly fast with machine learning. Connect your own ML models to your own data sources and query away!


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
Show an example of running on reviews or Tweets data

### Image Example (local directory)
Show an example of images in the local directory


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
- [How to connect ML APIs]() - Link to ReadMe
- [How to connect to Datastore]() - AWS S3 and all 
- 

## Contribute

We have many improvements we'd like to implement. Please help us! For the time being, please reach out to us directly using the form below if you'd like to help contribute.


## Contact Us

[Google Form Link]()
