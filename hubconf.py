"""
Ultrabrain CrowdFace Detector Model

Usage:
    import torch  
    model = torch.hub.load('Ultrabrain/CrowdFace', 'detector', autoshape=True)
"""
dependencies = [
    "gitpython",
    "ipython",
    "matplotlib>=3.2.2",
    "numpy>=1.18.5",
    "opencv-python>=4.1.1",
    "Pillow>=7.1.2",
    "psutil",
    "PyYAML>=5.3.1",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "thop>=0.1.1",
    "torch>=1.7.0",
    "torchvision>=0.8.1",
    "tqdm>=4.64.0",
    "ultralytics==8.0.136",
    "-e ."
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