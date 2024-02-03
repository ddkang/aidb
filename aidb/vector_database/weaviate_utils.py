from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WeaviateAuth:
  """
  :param url: weaviate url
  :param username: weaviate username
  :param pwd: weaviate password
  :param api_key: weaviate api key, user should choose input either username/pwd or api_key
  """
  url: Optional[str] = field(default=None)
  username: Optional[str] = field(default=None)
  pwd: Optional[str] = field(default=None)
  api_key: Optional[str] = field(default=None)