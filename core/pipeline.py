from typing import List, Optional

import numpy as np

from core.face_encoder import load_encodings
from core.face_matcher import find_matching_faces
from core.person_detector import PersonDetector, PersonDetection
from core.mosaic import apply_mosaic_to_bbox, apply_mosaic_to_mask


class Pipeline:
    def __init__(
        self,
        model_path: str,
        yolo_model: str = "yolov8n-seg.pt",
    ):
        """Initialize the face mosaic pipeline.

        Args:
            model_path: Path to .pkl file with face encodings.
            yolo_model: YOLOv8-seg model name.
        """
        self.known_encodings = load_encodings(model_path)
        self.detector = PersonDetector(model_name=yolo_model)

        # Configurable parameters (GUI can update these)
        self.face_tolerance: float = 0.5
        self.yolo_confidence: float = 0.5
        self.mosaic_block_size: int = 15
        self.use_segmentation: bool = True  # False = bbox mode

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect faces, match, blur matched persons.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Processed frame with mosaic applied to matched persons.
        """
        # Step 1: Find faces matching the known encodings
        matched_face_locs = find_matching_faces(
            frame, self.known_encodings, tolerance=self.face_tolerance
        )

        if not matched_face_locs:
            return frame

        # Step 2: Detect all persons with YOLOv8-seg
        person_detections = self.detector.detect(frame, confidence=self.yolo_confidence)

        if not person_detections:
            return frame

        # Step 3: Match faces to person detections and apply mosaic
        result = frame
        for detection in person_detections:
            if self._face_in_person(matched_face_locs, detection):
                if self.use_segmentation:
                    result = apply_mosaic_to_mask(
                        result, detection.mask, self.mosaic_block_size
                    )
                else:
                    result = apply_mosaic_to_bbox(
                        result, detection.bbox, self.mosaic_block_size
                    )

        return result

    def _face_in_person(
        self,
        face_locations: list,
        person: PersonDetection,
    ) -> bool:
        """Check if any matched face center falls within the person's bbox."""
        px1, py1, px2, py2 = person.bbox

        for top, right, bottom, left in face_locations:
            face_cx = (left + right) // 2
            face_cy = (top + bottom) // 2

            if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
                return True

        return False
