import os
import time

from aidb_utilities.blob_store.blob_store import DocumentType


def get_file_extension(file: str):
  return file.split('.')[-1]


def get_local_file_creation_time(file: str):
  return time.ctime(os.path.getctime(file))


def is_image_file(file: str):
  image_extension_filter = ['jpg', 'jpeg', 'png']
  file_ext = get_file_extension(file)
  return file_ext in image_extension_filter


def is_document(file: str):
  document_extension_filter = ['doc', 'docx', 'pdf']
  file_ext = get_file_extension(file)
  return file_ext in document_extension_filter


def get_document_type(file: str):
  file_ext = get_file_extension(file)
  if file_ext == 'pdf':
    return DocumentType.PDF
  elif file_ext == 'doc':
    return DocumentType.DOC
  elif file_ext == 'docx':
    return DocumentType.DOCX
  else:
    raise Exception("Unsupported document type")
