from typing import List

import boto3

from aidb.blob_store.blob_store import Blob, BlobStore, DocumentBlob, ImageBlob
from aidb.blob_store.utils import get_document_type, is_document, is_image_file


class AwsS3BlobStore(BlobStore):

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    session = boto3.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)
    s3 = session.resource('s3')
    self.bucket = s3.Bucket(bucket_name)

  def get_blobs(self) -> List[Blob]:
    pass


class AwsS3ImageBlobStore(AwsS3BlobStore):

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    super().__init__(bucket_name, access_key_id, secret_access_key)

  def get_blobs(self) -> List[ImageBlob]:
    image_blobs = []
    image_count = 0
    for obj in self.bucket.objects.all():
      file_path = f"s3://{obj.bucket_name}/{obj.key}"
      if is_image_file(file_path):
        file_creation_time = str(obj.last_modified)
        image_count += 1
        image_blobs.append(ImageBlob(blob_id=image_count, image_path=file_path, created_at=file_creation_time))
    return image_blobs


class AwsS3DocumentBlobStore(AwsS3BlobStore):

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    super().__init__(bucket_name, access_key_id, secret_access_key)

  def get_blobs(self) -> List[DocumentBlob]:
    doc_blobs = []
    doc_count = 0
    for obj in self.bucket.objects.all():
      file_path = f"s3://{obj.bucket_name}/{obj.key}"
      if is_document(file_path):
        file_creation_time = str(obj.last_modified)
        doc_type = get_document_type(file_path)
        doc_count += 1
        doc_blobs.append(
          DocumentBlob(blob_id=doc_count, doc_path=file_path, created_at=file_creation_time, doc_type=doc_type))
    return doc_blobs
