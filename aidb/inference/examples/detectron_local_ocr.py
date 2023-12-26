"""
Input: document segmentation model weights (model_final.pth), csv with paths to pdfs.

Output: csv with OCR'd text.
"""
import numpy as np
import pandas as pd

try:
  import pdf2image
except ImportError:
  import subprocess
  subprocess.run(["pip", "install", "pdf2image", "pytesseract", "detectron2"])
finally:
  import pdf2image
  import pytesseract
  from detectron2 import model_zoo
  from detectron2.engine import DefaultPredictor
  from detectron2.config import get_cfg

from aidb.inference.cached_inference_service import CachedInferenceService


class DetectronLocalOCR(CachedInferenceService):
  def __init__(
      self,
      name: str,
      model_path: str,
      device: str="cuda",
      is_single: bool=False,
  ):
    super().__init__(name=name, preferred_batch_size=1, is_single=is_single)
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.MAX_ITER = 300
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = model_path
    cfg.TEST.DETECTIONS_PER_IMAGE = 1
    self.od_model = DefaultPredictor(cfg)


  def signature(self):
    raise NotImplementedError()


  def infer_one(self, input: pd.Series) -> pd.DataFrame:
    pdf_path = list(input.to_dict().values())[0]
    im = np.array(pdf2image.convert_from_path(pdf_path, size=(2538, 3334))[0])
    outputs = self.od_model(im)
    boxes = list(outputs['instances'].pred_boxes)
    ocr_text = ''
    if len(boxes) > 0:
      box = boxes[0].detach().cpu().numpy()
      x0, y0, x1, y1 = box.astype(int)
      margin_x0 = max(0, x0-30)
      margin_x1 = min(im.shape[1], x1+30)
      margin_y0 = max(0, y0-30)
      margin_y1 = min(im.shape[0], y1+30)
      cropped_im = im[margin_y0:margin_y1, margin_x0:margin_x1]
      ocr_text = pytesseract.image_to_string(cropped_im)
    return pd.DataFrame([{'ocr_text': ocr_text}])


  def infer_batch(self, inputs: pd.DataFrame) -> pd.DataFrame:
    return super().infer_batch(inputs)
