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
from yolov9.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov9.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
import torch
import platform
from pathlib import Path

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

  bs = 1 
  view_img = check_imshow(warn=True)
  dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
  bs = len(dataset)
  model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  
  seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
  for path, im, im0s, _, _ in dataset:

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

    for i, det in enumerate(pred): 
      seen += 1
      p, im0, _ = path[i], im0s[i].copy(), dataset.count

      p = Path(p)  
      if len(det):
        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

      if view_img:
        if platform.system() == 'Linux' and p not in windows:
          windows.append(p)
          cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
          cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        cv2.waitKey(1) 

