# Connecting to Data Stores

We provide utilities to connect to different forms of data stores.
You can also implement your own.


## Images stored in the local storage

In our first example, we show how to access images stored in local storage:

```python
local_image_store = LocalImageBlobStore(data_dir)
image_blobs = local_image_store.get_blobs()
base_table_setup = BaseTablesSetup(DB_URL)
base_table_setup.insert_blob_meta_data('blob00', input_blobs, ['blob_id'])
```



## Documents stored in the AWS S3 storage

We also show how to access documents stored in S3:

```python
aws_doc_store = AwsS3DocumentBlobStore('bucket-name', '<your-aws-access-key>', 'your-secret-key')
doc_blobs = aws_doc_store.get_blobs()
base_table_setup = BaseTablesSetup(DB_URL)
base_table_setup.insert_blob_meta_data('blob00', doc_blobs, ['blob_id'])
```
