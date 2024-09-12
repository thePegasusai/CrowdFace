dependencies = [
    "torch",
    "torchvision",
]
import torch
from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend

def crowdface(auto_shape=True):
    """
    Load the CrowdFace model (without opencv dependency).
    """
    url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt"
    model = _DetectMultiBackend(weights=url)
    if auto_shape:
        model = _AutoShape(model)
    return model 
