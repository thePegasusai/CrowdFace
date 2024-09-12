dependencies = [
    "torch",
    "torchvision",
   # "opencv-python-headless",
]

import torch
import cv2
import numpy as np
from pathlib import Path
from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
from yolov9.utils.general import check_img_size, non_max_suppression

def crowdface(auto_shape=True):
    """
    Load the CrowdFace model with enhanced face detection and processing capabilities.

    Arguments:
      auto_shape (bool): Automatically adjust input shapes.
    Returns:
      CrowdFace model with advanced functionalities.
    """
    url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt"
    model = _DetectMultiBackend(weights=url)
    if auto_shape:
        model = _AutoShape(model)
    return model
