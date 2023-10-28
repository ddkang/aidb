from pathlib import Path
from typing import List

from aidb_utilities.blob_store.blob_store import Blob

from aidb_utilities.blob_store.blob_store import BlobStore, DocumentBlob, ImageBlob
from aidb_utilities.blob_store.utils import get_document_type, get_local_file_creation_time, is_document, is_image_file


class LocalBlobStore(BlobStore):

  def __init__(self, local_dir):
    self.files = Path(local_dir).rglob('*')

  def get_blobs(self) -> List[Blob]:
    pass


class LocalImageBlobStore(LocalBlobStore):

  def __init__(self, local_dir):
    super().__init__(local_dir)

  def get_blobs(self) -> List[ImageBlob]:
    image_blobs = []
    image_count = 0
    for file in self.files:
      file_path = str(file)
      if is_image_file(file_path):
        file_creation_time = get_local_file_creation_time(file_path)
        image_count += 1
        image_blobs.append(ImageBlob(blob_id=image_count, image_path=file_path, created_at=file_creation_time))
    return image_blobs


class LocalDocumentBlobStore(LocalBlobStore):

  def __init__(self, local_dir):
    super().__init__(local_dir)

  def get_blobs(self) -> List[DocumentBlob]:
    doc_blobs = []
    doc_count = 0
    for file in self.files:
      file_path = str(file)
      if is_document(file_path):
        file_creation_time = get_local_file_creation_time(file_path)
        doc_type = get_document_type(file_path)
        doc_count += 1
        doc_blobs.append(
          DocumentBlob(blob_id=doc_count, doc_path=file_path, created_at=file_creation_time, doc_type=doc_type))
    return doc_blobs
