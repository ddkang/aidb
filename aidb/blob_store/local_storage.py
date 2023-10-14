import glob

from typing import List

from aidb.blob_store.blob_store import ImageBlob, BlobStore, DocumentBlob
from blob_store.blob_store import Blob


class LocalBlobStore(BlobStore):

  def __init__(self, local_dir):
    self.files = glob.glob(local_dir, recursive=True)

  def get_blobs(self) -> List[Blob]:
    pass


class LocalImageBlobStore(LocalBlobStore):
  image_extension_filter = ['.jpg', '.jpeg', '.png']

  def __init__(self, local_dir):
    super().__init__(local_dir)

  def get_blobs(self) -> List[ImageBlob]:
    image_blobs = []
    for k, obj in self.files:
      image_blobs.append(ImageBlob(blob_id=k, image_path="", created_at=""))


class LocalDocumentBlobStore(LocalBlobStore):
  document_extension_filter = ['.doc', '.docx', '.pdf']

  def __init__(self, bucket_name, access_key_id, secret_access_key):
    super().__init__(bucket_name, access_key_id, secret_access_key)

  def get_blobs(self) -> List[DocumentBlob]:
    doc_blobs = []
    for k, obj in self.files:
      doc_blobs.append(ImageBlob(blob_id=k, image_path="", created_at=""))
