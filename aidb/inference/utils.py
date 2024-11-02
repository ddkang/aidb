import json
import logging
from typing import Any, List, Type

from pydantic import BaseModel, create_model

JSON_SCHEMA = 'json'
PYDANTIC_SCHEMA = 'pydantic'


def create_pydantic_model(model_name: str, fields: dict, is_single: bool) -> Type[BaseModel]:
  """
    Creates a Pydantic model for structured output based on the provided fields.

    Args:
        model_name (str): The name of the model to be created.
        fields (dict): A dictionary where keys are field names and values are field types as strings.
        is_single (bool): A flag indicating if the model should be a single instance or a list.

    Returns:
        Type[BaseModel]: A Pydantic model class.
    """
  type_mapping = {
      'str': str,
      'int': int,
      'float': float,
      'bool': bool,
      # Add more types as needed
  }
  model_fields = {}

  for name, type_str in fields.items():
    python_type = type_mapping.get(type_str, Any)
    if python_type is Any:
      logging.warning(f"Warning: Unsupported type '{type_str}' for field '{name}'. Defaulting to 'Any'.")
    model_fields[name] = (python_type, ...)  # Default value

  model = create_model(
      model_name,
      **model_fields,
  )
  if not is_single:
    model = create_model(
        f'{model_name}_list',
        items=(List[model], (...)),
    )
  return model


def parse_output_schema(service_name, output_schema_config: dict, is_single: bool):
  """
    Parses the output schema configuration and returns the appropriate schema.

    Args:
        service_name (str): The name of the service for which the schema is being parsed.
        output_schema_config (dict): A dictionary containing the schema configuration.
        is_single (bool): A flag indicating if the schema should be for a single instance or a list.

    Returns:
        dict or Type[BaseModel]: The parsed output schema, either as a JSON schema dictionary or a Pydantic model.
    """
  if JSON_SCHEMA in output_schema_config:
    with open(output_schema_config[JSON_SCHEMA], 'r') as f:
      output_schema = json.load(f)
    output_schema = {
        'type': 'json_schema',
        'json_schema': output_schema,
    }
  elif PYDANTIC_SCHEMA in output_schema_config:
    output_schema = create_pydantic_model(
        service_name, output_schema_config[PYDANTIC_SCHEMA], is_single)
  else:
    raise Exception('Invalid output schema configuration')

  return output_schema
