import json
from typing import Dict, List, Tuple, Union
import logging
import pandas as pd
from jinja2 import Template
from litellm import completion

from aidb.inference.http_inference_service import CachedInferenceService
from aidb.inference.utils import parse_output_schema
from aidb.utils.perf_utils import call_counter


class LLMInference(CachedInferenceService):
  def __init__(
      self,
      *args,
      model: str = None,
      output_schema: Dict[str, str] = None,
      prompt: str = None,
      model_config: dict = None,
      **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.model = model
    self.template = Template(prompt)
    self.output_schema = parse_output_schema(
        self.name, output_schema, self.is_single)
    self.model_config = model_config

  def signature(self) -> Tuple[List, List]:
    raise NotImplementedError()

  def _convert_response_to_output(self, response) -> pd.DataFrame:
    """
    Converts the model response to a pandas DataFrame.

    Args:
        response: The response from the model.

    Returns:
        pd.DataFrame: The parsed response as a DataFrame.

    Raises:
        ValueError: If the response content cannot be parsed.
    """
    try:
      response = response['choices'][0]['message']['content']
      json_content = response['choices'][0]['message']['content']
      parsed_content = json.loads(json_content)
    except (KeyError, json.JSONDecodeError) as e:
      raise Exception("Failed to parse response: %s", response)

    if self.is_single:
      df = pd.DataFrame([parsed_content])
    else:
      dfs = [pd.DataFrame(value) for value in parsed_content.values()]
      df = pd.concat(dfs, axis=1)
    return df

  @call_counter
  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    """
    Performs inference on a single input.

    Args:
        input (pd.Series): The input data as a pandas Series.

    Returns:
        pd.DataFrame: The inference results as a DataFrame.
    """
    context = {key.replace('.', '_'): value for key, value in input.items()}
    data = self.template.render(**context)

    try:
      response = completion(
          model=self.model,
          messages=[{"content": data, "role": "user"}],
          response_format=self.output_schema,
          **self.model_config
      )
    except Exception as e:
      logging.error("Inference failed: %s", e)
      raise
    response = '{"summary":"The reviewer praises Amazon\'s batteries for'
    inference_results = self._convert_response_to_output(response)
    return inference_results
