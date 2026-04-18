from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class 0 = person


@dataclass
class PersonDetection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    mask: np.ndarray  # Binary mask, same size as input frame
    confidence: float


class PersonDetector:
    def __init__(self, model_name: str = "yolov8n-seg.pt"):
        """Initialize with a YOLOv8 segmentation model.

        Args:
            model_name: Model variant. 'yolov8n-seg.pt' (nano, fastest)
                        or 'yolov8s-seg.pt' (small, more accurate).
        """
        self.model = YOLO(model_name)

    def detect(
        self, frame: np.ndarray, confidence: float = 0.5
    ) -> List[PersonDetection]:
        """Detect persons in frame, return bboxes and segmentation masks.

        Args:
            frame: BGR image as numpy array.
            confidence: Minimum confidence threshold.

        Returns:
            List of PersonDetection with bbox, mask, and confidence.
        """
        results = self.model(frame, conf=confidence, classes=[PERSON_CLASS_ID], verbose=False)
        result = results[0]

        detections = []
        if result.masks is None or len(result.boxes.cls) == 0:
            return detections

        h, w = frame.shape[:2]

        for i, cls_id in enumerate(result.boxes.cls):
            if int(cls_id) != PERSON_CLASS_ID:
                continue

            bbox_raw = result.boxes.xyxy[i]
            bbox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
            conf = float(result.boxes.conf[i])

            mask_data = result.masks.data[i]
            if hasattr(mask_data, "cpu"):
                mask = mask_data.cpu().numpy().astype(np.uint8)
            else:
                mask = np.array(mask_data).astype(np.uint8)

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            detections.append(PersonDetection(bbox=bbox, mask=mask, confidence=conf))

        return detections
