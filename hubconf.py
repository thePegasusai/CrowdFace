import torch
import cv2
import numpy as np
from pathlib import Path

# Ensure the correct versions of dependencies are installed
dependencies = [
    "torch",
    "torchvision",
    "opencv-python-headless",  # OpenCV is required for image processing
    "yolov9"  # Ensure yolov9 is installed
]

# Try to import yolov9 modules
try:
    from yolov9.models.common import AutoShape as _AutoShape
    from yolov9.models.common import DetectMultiBackend as _DetectMultiBackend
    from yolov9.utils.general import check_img_size, non_max_suppression
except ImportError as e:
    print(f"Error importing yolov9 modules: {e}")
    # Handle the error or install the necessary packages

# Define the enhanced detector with face replacement capabilities
def crowdface(auto_shape=True):
    """
    Load the CrowdFace model with enhanced face detection and processing capabilities.
    
    Arguments:
      auto_shape (bool): Automatically adjust input shapes.
    Returns:
      CrowdFaceModel instance with advanced functionalities.
    """
    url = "https://github.com/Ultrabrain/CrowdFace/releases/download/yolov9FaceRecognition/best.pt"
    model = _DetectMultiBackend(weights=url)
    if auto_shape:
        model = _AutoShape(model)
    return CrowdFaceModel(model)

class CrowdFaceModel:
    def __init__(self, model):
        self.model = model
        # Pre-load or define any resources needed for face processing (e.g., overlay images)

    def process_frame(self, frame, overlay_img_path, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic_nms=False):
        # Load the overlay image
        overlay_img = cv2.imread(overlay_img_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            raise ValueError(f"Overlay image not found at path: {overlay_img_path}")
        overlay_img = self.prepare_overlay_image(overlay_img)

        img = torch.from_numpy(frame).to(self.model.device)
        img = img.float() / 255.0  # Normalize image
        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension

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
        """
        Apply face replacement by blending an overlay image with a detected face region.
        
        Arguments:
          frame: The original frame.
          xyxy: The coordinates of the detected face.
          overlay_img: The overlay image for face replacement.
        """
        x1, y1, x2, y2 = xyxy
        face_region = frame[y1:y2, x1:x2]

        # Apply Gaussian blur for privacy enhancement
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
        
        # Blend the blurred face with the original face region
        alpha = 0.5  # Adjust alpha between 0 to 1 for blending ratio
        blended_face = cv2.addWeighted(face_region, 1 - alpha, blurred_face, alpha, 0)
        
        # Replace the original face region with the blended one
        frame[y1:y2, x1:x2] = blended_face
        
        # Seamlessly clone the overlay image onto the face region
        center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
        frame = cv2.seamlessClone(overlay_img, frame, np.full(overlay_img.shape[:2], 255, dtype=np.uint8), center, cv2.MIXED_CLONE)

        return frame  # Return the modified frame

# Load the model with force_reload to ensure the latest version
model = torch.hub.load('Ultrabrain/CrowdFace:main', 'crowdface', autoshape=True, force_reload=True)
