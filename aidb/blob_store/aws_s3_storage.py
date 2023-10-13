from typing import List

import boto3
from aidb.blob_store.blob_store import ImageBlob, BlobStore, DocumentBlob
from blob_store.blob_store import Blob


class AwsS3BlobStore(BlobStore):

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    session = boto3.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    s3 = session.resource('s3')
    self.bucket = s3.Bucket(bucket_name)

  def get_blobs(self) -> List[Blob]:
    pass


class AwsS3ImageBlobStore(AwsS3BlobStore):
  image_extension_filter = ['.jpg', '.jpeg', '.png']

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    super().__init__(bucket_name, access_key_id, secret_access_key)

  def get_blobs(self) -> List[ImageBlob]:
    image_blobs = []
    for k, obj in self.bucket.objects.all():
      image_blobs.append(ImageBlob(blob_id=k, image_path="", created_at=""))


class AwsS3DocumentBlobStore(AwsS3BlobStore):
  document_extension_filter = ['.doc', '.docx', '.pdf']

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    super().__init__(bucket_name, access_key_id, secret_access_key)

  def get_blobs(self) -> List[DocumentBlob]:
    doc_blobs = []
    for k, obj in self.bucket.objects.all():
      doc_blobs.append(ImageBlob(blob_id=k, image_path="", created_at=""))
