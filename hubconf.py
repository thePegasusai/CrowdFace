"""
Ultrabrain CrowdFace Detector Model

Usage:
    import torch  
    model = torch.hub.load('Ultrabrain/CrowdFace', 'detector', autoshape=True)
"""
dependencies = [
    "torch",
    "torchvision",
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

  url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt" 
  backup = "https://ultrabrain.s3.amazonaws.com/best.pt"
  model = _DetectMultiBackend(weights=url, backup=backup)
  if auto_shape: model = _AutoShape(model)
  return model

def detect(source=0, *args, **kwargs):
  url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt" 
  backup = "https://ultrabrain.s3.amazonaws.com/best.pt"
  
  _run(*args, **kwargs, model=url, source=source)
