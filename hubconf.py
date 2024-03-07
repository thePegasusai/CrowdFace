"""
Ultrabrain CrowdFace Detector Model

Usage:
    import torch  
    model = torch.hub.load('Ultrabrain/CrowdFace', 'detector', autoshape=True)
"""
dependencies = [
    "torch",
    "torchvision",
    "ultralytics"
]

from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
from yolov9.detect import run as _run

def detector(auto_shape=True):
  """
  Load CrowdFace detecting YOLOv9 model
  
  Arguments:
    auto_shape (bool): apply YOLO .autoshape() wrapper to model 
  Returns:
    CrowdFace detecting YOLOv9 model
  """

  url = "https://ultrabrain.s3.amazonaws.com/best.pt"
  model = _DetectMultiBackend(url)
  if auto_shape: model = _AutoShape(model)
  return model

def detect(*args, **kwargs):
  _run(*args, **kwargs)