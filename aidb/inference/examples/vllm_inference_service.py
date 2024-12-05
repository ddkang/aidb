import json
from typing import Dict, List, Tuple, Union
import logging
import pandas as pd
from jinja2 import Template
from litellm import completion

from aidb.inference.http_inference_service import CachedInferenceService
from aidb.inference.utils import parse_output_schema
from aidb.utils.perf_utils import call_counter
from transformers import pipeline

import torch
from aidb.utils.timer import Timer
from vllm import LLM, SamplingParams


class VLLMInference(CachedInferenceService):
  def __init__(
      self,
      *args,
      model_name: str = None,
      prompt: str = None,
      model_config: dict = None,
      sampling_params: dict = None,
      apply_chat_template: bool = False,
      **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.template = Template(prompt)
    self.model_config = model_config
    self.model_name = model_name
    self.llm = None
    self.timer = Timer()
    if sampling_params:
      self.sampling_params = SamplingParams(**sampling_params)
    else:
      self.sampling_params = None
    self.apply_chat_template = apply_chat_template


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
      json_content = response[0]['generated_text']
      for item in json_content:
        if item['role'] == 'assistant':
          parsed_content = item['content']
    except (KeyError, json.JSONDecodeError) as e:
      raise Exception("Failed to parse response: %s", response)

    if self.is_single:
      df = pd.DataFrame([parsed_content])
    else:
      dfs = [pd.DataFrame(value) for value in parsed_content.values()]
      df = pd.concat(dfs, axis=1)
    return df


  def _load_model(self):
    if self.llm is None:
      self.timer.start()
      if self.model_config:
        self.llm = LLM(model=self.model_name, **self.model_config)
      else:
        self.llm = LLM(model=self.model_name)
      logging.info(f"{self.model_name} model loading time: {self.timer.check(f'{self.model_name} model loading')}")

  def _convert_input_data(self, input):
    context = {key.replace('.', '_'): value for key, value in input.items()}
    message = self.template.render(**context)
    if self.apply_chat_template:
      tokenizer = self.llm.get_tokenizer()
      message = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': message}],
        tokenize=False,
      )
    return message

    
  @call_counter
  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    """
    Performs inference on a single input.

    Args:
        input (pd.Series): The input data as a pandas Series.

    Returns:
        pd.DataFrame: The inference results as a DataFrame.
    """
    self._load_model()
    message = [self._convert_input_data(input)]
    try:
      response = self.llm.generate(message, sampling_params= self.sampling_params)
    except Exception as e:
      logging.error("Inference failed: %s", e)
      raise
    inference_result = []
    for output in response:
      inference_result.append(output.outputs[0].text)
    return pd.DataFrame(inference_result)
  

  def infer_batch(self, inputs: pd.DataFrame) -> pd.DataFrame:
    self._load_model()
    messages = []
    for _, row in inputs.iterrows():
      messages.append(self._convert_input_data(row))
    try:
      responses = self.llm.generate(messages, sampling_params= self.sampling_params)
    except Exception as e:
      logging.error("Inference failed: %s", e)
      raise
    inference_results = []
    for output in responses:
      output_group = []
      for item in output.outputs:
        text = item.text
        if self.apply_chat_template:
          text = text.replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '')
        output_group.append(text)
      inference_results.append(pd.DataFrame(output_group))

    return inference_results