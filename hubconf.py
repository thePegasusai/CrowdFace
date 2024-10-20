"""
Ultrabrain CrowdFace Detector Model
CrowdFace: Advanced Face Detection and Replacement
Leverages YOLOv9 for efficient and accurate face detection, integrating seamless face replacement and privacy enhancements.
Usage:
    import torch  
    model = torch.hub.load('Ultrabrain/CrowdFace', 'detector', autoshape=True)
    model = torch.hub.load('Ultrabrain/CrowdFace', 'crowdface', autoshape=True)
"""
dependencies = [
    "torch",
    "torchvision",
    "opencv-python-headless",  # OpenCV is required for image processing
]

from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
from yolov9.detect import run as _run
from yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import torch
import platform
import cv2
import numpy as np
from pathlib import Path
from yolov9.models.common import AutoShape as _AutoShape
from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
from yolov9.utils.general import check_img_size, non_max_suppression

def detector(auto_shape=True):
  """
  Load CrowdFace detecting YOLOv9 model
  
  Arguments:
    auto_shape (bool): apply YOLO .autoshape() wrapper to model 
  Returns:
    CrowdFace detecting YOLOv9 model
  """
# Define the enhanced detector with face replacement capabilities
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
    return CrowdFaceModel(model)

  url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt" 
  backup = "https://ultrabrain.s3.amazonaws.com/best.pt"
  model = _DetectMultiBackend(weights=url, backup=backup)
  if auto_shape: model = _AutoShape(model)
  return model
class CrowdFaceModel:
    def __init__(self, model):
        self.model = model
        # Pre-load or define any resources needed for face processing (e.g., overlay images)

def detect(
    model,
    source=0,
    imgsz=(640, 640),
    conf_thres=0.25, 
    iou_thres=0.45,  
    max_det=1000,  
    device='',  
    view_img=False,  
    classes=None,  
    agnostic_nms=False,  
    augment=False,  
    visualize=False,  
    vid_stride=1,
):
  stride, _, pt = model.stride, model.names, model.pt
  imgsz = check_img_size(imgsz, s=stride)
    def process_frame(self, frame, overlay_img_path, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
        """
        Process a single frame for face detection and apply face replacement/enhancements.
        
        Arguments:
          frame: Frame to process.
          overlay_img_path: Path to the overlay image for face replacement.
          conf_thres: Confidence threshold for detections.
          iou_thres: IOU threshold for non-max suppression.
          classes: Target classes for detection.
          agnostic_nms: Apply class-agnostic NMS.
        """
        # Load the overlay image
        overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        overlay_img = self.prepare_overlay_image(overlay_img)

  bs = 1 
  view_img = check_imshow(warn=True)
  dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
  bs = len(dataset)
  model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  
  seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
  for path, im, im0s, _, _ in dataset:
        img = torch.from_numpy(frame).to(self.model.device)
        img = img.float() / 255.0  # Normalize image
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

    with dt[0]:
      im = torch.from_numpy(im).to(model.device)
      im = im.half() if model.fp16 else im.float()  
      im /= 255  
      if len(im.shape) == 3:
        im = im[None] 
    
    with dt[1]:
      pred = model(im, augment=augment, visualize=visualize)
    
    with dt[2]:
      pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # Perform detection
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)
        # Apply face replacement or enhancements on detections
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    self.apply_face_replacement(frame, list(map(int, xyxy)), overlay_img)
        return frame
    def prepare_overlay_image(self, overlay_img):
        # Implement any preprocessing needed for the overlay image
        return overlay_img
    def apply_face_replacement(self, frame, xyxy, overlay_img):
        # Example: Gaussian blur for privacy enhancement
        x1, y1, x2, y2 = xyxy
        face_region = frame[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        frame[y1:y2, x1:x2] = blurred_face

    for i, det in enumerate(pred): 
      seen += 1
      p, im0, _ = path[i], im0s[i].copy(), dataset.count
        # Example: Seamless cloning for overlaying an image onto the face region
        center = (x1 + (x2-x1)//2, y1 + (y2-y1)//2)
        frame = cv2.seamlessClone(overlay_img, frame, np.full(overlay_img.shape[:2], 255, dtype=np.uint8), center, cv2.MIXED_CLONE)

      p = Path(p)  
      if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
# Additional customization and utility functions can be added to the CrowdFaceModel class.

      if view_img:
        if platform.system() == 'Linux' and p not in windows:
          windows.append(p)
          cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
          cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1) 
