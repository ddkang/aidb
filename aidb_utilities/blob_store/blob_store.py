from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Union


class DocumentType(Enum):
  PDF = 'pdf'
  DOCX = 'docx'
  DOC = 'doc'


@dataclass
class Blob:
  blob_id: int

  def to_dict(self):
    return {
      'blob_id': self.blob_id,
    }


@dataclass
class ImageBlob(Blob):
  image_path: str
  created_at: Union[None, str]

  def to_dict(self):
    return {
      'blob_id': self.blob_id,
      'image_path': self.image_path,
      'created_at': self.created_at
    }


@dataclass
class DocumentBlob(Blob):
  doc_path: str
  created_at: str
  doc_type: DocumentType

  def to_dict(self):
    return {
      'blob_id': self.blob_id,
      'doc_path': self.doc_path,
      'created_at': self.created_at,
      'doc_type': self.doc_type.value
    }


class BlobStore(ABC):

  def __int__(self):
    '''
    configuration, data store access keys, etc.
    '''
    pass

  @abstractmethod
  def get_blobs(self) -> List[Blob]:
    pass
