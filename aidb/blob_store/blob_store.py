from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Union
from enum import Enum


class DocumentType(Enum):
  PDF = "pdf"
  DOCX = "docx"
  DOC = "doc"


@dataclass
class Blob:
  blob_id: int


@dataclass
class ImageBlob(Blob):
  image_path: str
  created_at: Union[None, str]


@dataclass
class DocumentBlob(Blob):
  doc_path: str
  created_at: str
  doc_type: DocumentType


class BlobStore(ABC):

  def __int__(self):
    """
    configuration, data store access keys, etc.
    """
    pass

  @abstractmethod
  def get_blobs(self) -> List[Blob]:
    pass
