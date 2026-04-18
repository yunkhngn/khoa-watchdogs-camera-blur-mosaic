from typing import List, Optional

import cv2
import numpy as np

from core.face_encoder import load_encodings
from core.face_matcher import find_matching_faces
from core.person_detector import PersonDetector, PersonDetection
from core.mosaic import apply_mosaic_to_bbox, apply_mosaic_to_mask, draw_hacker_box
from core.camera_overlay import apply_cctv_overlay


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
        self.grayscale_cam: bool = True     # True = B&W camera view

        # --- Performance: face detection cache ---
        self._frame_count: int = 0
        self._face_detect_interval: int = 5  # Run face detection every N frames
        self._cached_face_locs: list = []    # Cache matched face locations
        self._face_scale: float = 0.5        # Downscale factor for face detection
        self._has_match: bool = False         # Whether we found a match recently

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame: detect faces, match, blur matched persons.

        Optimized: face detection runs every N frames on a downscaled image,
        while person segmentation runs every frame for smooth tracking.

        Args:
            frame: BGR image as numpy array.

        Returns:
            Processed frame with mosaic applied to matched persons.
        """
        if self.grayscale_cam:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        self._frame_count += 1
        result = frame

        # Step 1: Run face detection periodically (it's the slow part)
        if self._frame_count % self._face_detect_interval == 1 or not self._has_match:
            self._update_face_cache(frame)

        # Step 2: Detect persons and apply effects if there's a match
        if self._has_match:
            person_detections = self.detector.detect(frame, confidence=self.yolo_confidence)
            
            if person_detections:
                for detection in person_detections:
                    if self._face_in_person(self._cached_face_locs, detection):
                        if self.use_segmentation:
                            result = apply_mosaic_to_mask(
                                result, detection.mask, self.mosaic_block_size
                            )
                        else:
                            result = apply_mosaic_to_bbox(
                                result, detection.bbox, self.mosaic_block_size
                            )
                        
                        # Hacker cyberpunk overlay!
                        draw_hacker_box(result, detection.bbox)

        # Step 3: Apply global CCTV camera style overlay to EVERY frame
        result = apply_cctv_overlay(result)
        
        return result

    def _update_face_cache(self, frame: np.ndarray) -> None:
        """Run face detection on a downscaled frame and cache results."""
        h, w = frame.shape[:2]
        scale = self._face_scale

        # Downscale for faster face detection
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))

        matched_face_locs_small = find_matching_faces(
            small, self.known_encodings, tolerance=self.face_tolerance
        )

        if matched_face_locs_small:
            # Scale face locations back to original frame size
            inv = 1.0 / scale
            self._cached_face_locs = [
                (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
                for top, right, bottom, left in matched_face_locs_small
            ]
            self._has_match = True
        else:
            self._cached_face_locs = []
            self._has_match = False

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
