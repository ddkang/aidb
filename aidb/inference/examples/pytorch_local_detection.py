import pandas as pd

from groundingdino.util.inference import Model
from aidb.inference.cached_inference_service import CachedInferenceService


class PyTorchLocalObjectDetection(CachedInferenceService):
  def __init__(
      self,
      name: str,
      model_config_path: str,
      model_checkpoint_path: str,
      caption: str,
      is_single: bool=False,
      use_batch: bool=True,
      batch_size: int=1,
      box_threshold: float=0.35,
      device: str="cuda",
  ):
    super().__init__(name=name, preferred_batch_size=batch_size, is_single=is_single)
    self._model = Model(model_config_path, model_checkpoint_path, device=device)
    self._caption = caption
    self._use_batch = use_batch
    self._box_threshold = box_threshold


  def signature(self):
    raise NotImplementedError()


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    image = [list(input.to_dict().values())[0]]
    output = self._model.predict_with_caption(image, self._caption, self._box_threshold)[0]
    output = [
      {
        "min_x": xyxy[0],
        "min_y": xyxy[1],
        "max_x": xyxy[2],
        "max_y": xyxy[3],
        "confidence": conf,
      } for xyxy, conf in zip(output.xyxy, output.confidence)]
    return pd.DataFrame(output)


  def infer_batch(self, inputs: pd.DataFrame) -> pd.DataFrame:
    if not self._use_batch:
      return super().infer_batch(inputs)

    images = list(inputs.to_dict(orient="list").values())[0]
    outputs_merge = []
    for i in range(0, len(images), self.preferred_batch_size):
      image_batch = images[i:i+self.preferred_batch_size] if i+self.preferred_batch_size < len(images) else images[i:]
      outputs = self._model.predict_with_caption(image_batch, self._caption, self._box_threshold)
      outputs = [
        {
          "min_x": xyxy[0],
          "min_y": xyxy[1],
          "max_x": xyxy[2],
          "max_y": xyxy[3],
          "confidence": conf,
        } for output in outputs for xyxy, conf in zip(output.xyxy, output.confidence)]
      outputs_merge.extend(outputs)
    return pd.DataFrame(outputs_merge)
