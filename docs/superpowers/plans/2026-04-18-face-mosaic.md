# Face Mosaic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a PyQt6 desktop app that recognizes the user's face via pre-encoded embeddings, then applies mosaic blur to their entire body using YOLOv8 instance segmentation — leaving other people unblurred.

**Architecture:** A `core/` pipeline (face encoding → face matching → person segmentation → mosaic) orchestrated by `pipeline.py`, rendered through a PyQt6 GUI with threaded video/webcam workers. Core is fully decoupled from GUI for testability.

**Tech Stack:** Python 3.11+, face_recognition (dlib), ultralytics (YOLOv8-seg), PyQt6, OpenCV, NumPy, pytest

---

## File Structure

```
face-mosaic/
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── core/
│   ├── __init__.py
│   ├── face_encoder.py          # Encode folder → embeddings .pkl
│   ├── face_matcher.py          # Match faces in frame vs embeddings
│   ├── person_detector.py       # YOLOv8-seg person detection
│   ├── mosaic.py                # Apply mosaic blur
│   └── pipeline.py              # Orchestrate all modules
├── gui/
│   ├── __init__.py
│   ├── main_window.py           # Main window layout
│   ├── video_widget.py          # Video display widget
│   ├── controls_widget.py       # Input/settings/actions panels
│   └── workers.py               # QThread workers
├── data/
│   ├── reference_faces/         # User drops face photos here
│   └── models/                  # Encoded .pkl files
├── output/                      # Exported files
└── tests/
    ├── __init__.py
    ├── test_face_encoder.py
    ├── test_face_matcher.py
    ├── test_person_detector.py
    ├── test_mosaic.py
    └── test_pipeline.py
```

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `main.py`
- Create: all `__init__.py` files
- Create: all directories

- [ ] **Step 1: Initialize git repo and create directory structure**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
git init
mkdir -p core gui data/reference_faces data/models output tests
touch core/__init__.py gui/__init__.py tests/__init__.py
```

- [ ] **Step 2: Create requirements.txt**

```txt
face_recognition>=1.3.0
ultralytics>=8.1.0
PyQt6>=6.6.0
opencv-python>=4.9.0
numpy>=1.26.0
pytest>=8.0.0
```

- [ ] **Step 3: Create main.py entry point**

```python
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Face Mosaic")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Install dependencies**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
pip install -r requirements.txt
```

Expected: All packages install successfully. `face_recognition` may take a few minutes (compiles dlib).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: project setup with directory structure and dependencies"
```

---

## Task 2: Face Encoder Module

**Files:**
- Create: `core/face_encoder.py`
- Create: `tests/test_face_encoder.py`

- [ ] **Step 1: Write tests for face encoder**

```python
# tests/test_face_encoder.py
import os
import tempfile
import pickle
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.face_encoder import encode_faces_from_folder, save_encodings, load_encodings


class TestEncodeFacesFromFolder:
    def test_returns_list_of_numpy_arrays(self, tmp_path):
        """With valid face images, returns list of 128-d numpy arrays."""
        fake_encoding = np.random.rand(128)
        with patch("core.face_encoder.face_recognition") as mock_fr:
            # Create a dummy image file
            img_path = tmp_path / "face1.jpg"
            img_path.write_bytes(b"fake image data")

            mock_fr.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_fr.face_encodings.return_value = [fake_encoding]

            result = encode_faces_from_folder(str(tmp_path))

            assert len(result) == 1
            assert result[0].shape == (128,)
            np.testing.assert_array_equal(result[0], fake_encoding)

    def test_skips_images_with_no_faces(self, tmp_path):
        """Images with no detectable faces are skipped."""
        with patch("core.face_encoder.face_recognition") as mock_fr:
            img_path = tmp_path / "noface.jpg"
            img_path.write_bytes(b"fake image data")

            mock_fr.load_image_file.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_fr.face_encodings.return_value = []

            result = encode_faces_from_folder(str(tmp_path))
            assert len(result) == 0

    def test_empty_folder_returns_empty_list(self, tmp_path):
        result = encode_faces_from_folder(str(tmp_path))
        assert result == []

    def test_raises_on_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            encode_faces_from_folder("/nonexistent/path")


class TestSaveLoadEncodings:
    def test_roundtrip(self, tmp_path):
        """Save then load returns identical encodings."""
        encodings = [np.random.rand(128) for _ in range(5)]
        path = str(tmp_path / "model.pkl")

        save_encodings(encodings, path)
        loaded = load_encodings(path)

        assert len(loaded) == 5
        for orig, loaded_enc in zip(encodings, loaded):
            np.testing.assert_array_almost_equal(orig, loaded_enc)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_encodings("/nonexistent/model.pkl")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python -m pytest tests/test_face_encoder.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'core.face_encoder'`

- [ ] **Step 3: Implement face_encoder.py**

```python
# core/face_encoder.py
import os
import pickle
from typing import List

import numpy as np
import face_recognition

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_faces_from_folder(folder_path: str) -> List[np.ndarray]:
    """Scan folder for images, detect faces, return list of 128-d encodings."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    encodings = []
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        filepath = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(filepath)
        face_encs = face_recognition.face_encodings(image)

        if face_encs:
            # Take the first (largest) face from each image
            encodings.append(face_encs[0])

    return encodings


def save_encodings(encodings: List[np.ndarray], output_path: str) -> None:
    """Save encodings list to a pickle file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(encodings, f)


def load_encodings(model_path: str) -> List[np.ndarray]:
    """Load encodings list from a pickle file."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python -m pytest tests/test_face_encoder.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/face_encoder.py tests/test_face_encoder.py
git commit -m "feat: add face encoder module with save/load support"
```

---

## Task 3: Face Matcher Module

**Files:**
- Create: `core/face_matcher.py`
- Create: `tests/test_face_matcher.py`

- [ ] **Step 1: Write tests for face matcher**

```python
# tests/test_face_matcher.py
import numpy as np
import pytest
from unittest.mock import patch

from core.face_matcher import find_matching_faces


class TestFindMatchingFaces:
    def setup_method(self):
        # Deterministic "known" encoding
        self.known = [np.zeros(128)]

    def test_returns_matched_face_locations(self):
        """When a face matches, returns its (top, right, bottom, left) location."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        face_loc = (50, 200, 150, 100)  # top, right, bottom, left

        with patch("core.face_matcher.face_recognition") as mock_fr:
            mock_fr.face_locations.return_value = [face_loc]
            mock_fr.face_encodings.return_value = [np.zeros(128)]
            mock_fr.compare_faces.return_value = [True]

            result = find_matching_faces(frame, self.known, tolerance=0.6)

            assert len(result) == 1
            assert result[0] == face_loc

    def test_no_match_returns_empty(self):
        """When no face matches known encodings, returns empty list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("core.face_matcher.face_recognition") as mock_fr:
            mock_fr.face_locations.return_value = [(50, 200, 150, 100)]
            mock_fr.face_encodings.return_value = [np.ones(128)]
            mock_fr.compare_faces.return_value = [False]

            result = find_matching_faces(frame, self.known, tolerance=0.6)
            assert result == []

    def test_no_faces_in_frame(self):
        """When no faces detected at all, returns empty list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("core.face_matcher.face_recognition") as mock_fr:
            mock_fr.face_locations.return_value = []
            mock_fr.face_encodings.return_value = []

            result = find_matching_faces(frame, self.known)
            assert result == []

    def test_empty_known_encodings(self):
        """With no known encodings, returns empty list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = find_matching_faces(frame, [], tolerance=0.6)
        assert result == []

    def test_multiple_faces_only_matches_returned(self):
        """With multiple faces, only matching ones are returned."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        loc1 = (50, 200, 150, 100)
        loc2 = (50, 400, 150, 300)

        with patch("core.face_matcher.face_recognition") as mock_fr:
            mock_fr.face_locations.return_value = [loc1, loc2]
            mock_fr.face_encodings.return_value = [np.zeros(128), np.ones(128)]
            mock_fr.compare_faces.return_value = [True, False]

            result = find_matching_faces(frame, self.known, tolerance=0.6)
            assert result == [loc1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_face_matcher.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement face_matcher.py**

```python
# core/face_matcher.py
from typing import List, Tuple

import numpy as np
import face_recognition

FaceLocation = Tuple[int, int, int, int]  # (top, right, bottom, left)


def find_matching_faces(
    frame: np.ndarray,
    known_encodings: List[np.ndarray],
    tolerance: float = 0.6,
    model: str = "hog",
) -> List[FaceLocation]:
    """Detect faces in frame, return locations of those matching known encodings.

    Args:
        frame: BGR or RGB image as numpy array.
        known_encodings: List of 128-d face encoding vectors.
        tolerance: Distance threshold. Lower = stricter matching.
        model: 'hog' (faster, CPU) or 'cnn' (more accurate, GPU).

    Returns:
        List of (top, right, bottom, left) tuples for matched faces.
    """
    if not known_encodings:
        return []

    face_locations = face_recognition.face_locations(frame, model=model)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(frame, face_locations)

    matched_locations = []
    for location, encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=tolerance
        )
        if any(matches):
            matched_locations.append(location)

    return matched_locations
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_face_matcher.py -v
```

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/face_matcher.py tests/test_face_matcher.py
git commit -m "feat: add face matcher module"
```

---

## Task 4: Person Detector Module (YOLOv8-seg)

**Files:**
- Create: `core/person_detector.py`
- Create: `tests/test_person_detector.py`

- [ ] **Step 1: Write tests for person detector**

```python
# tests/test_person_detector.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from core.person_detector import PersonDetector, PersonDetection


class TestPersonDetector:
    def _make_mock_result(self, boxes_xyxy, masks_data, confs, cls_ids):
        """Helper to create a mock ultralytics result."""
        result = MagicMock()
        boxes = MagicMock()
        boxes.xyxy = np.array(boxes_xyxy) if boxes_xyxy else np.empty((0, 4))
        boxes.conf = np.array(confs) if confs else np.empty(0)
        boxes.cls = np.array(cls_ids) if cls_ids else np.empty(0)
        result.boxes = boxes

        if masks_data:
            masks = MagicMock()
            masks.data = np.array(masks_data)
            result.masks = masks
        else:
            result.masks = None

        return result

    @patch("core.person_detector.YOLO")
    def test_detect_returns_person_detections(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mask = np.ones((480, 640), dtype=np.uint8)
        mock_result = self._make_mock_result(
            boxes_xyxy=[[100, 50, 300, 400]],
            masks_data=[mask],
            confs=[0.9],
            cls_ids=[0],  # 0 = person in COCO
        )
        mock_model.return_value = [mock_result]

        detector = PersonDetector()
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))

        assert len(detections) == 1
        assert detections[0].bbox == (100, 50, 300, 400)
        assert detections[0].confidence == pytest.approx(0.9)
        np.testing.assert_array_equal(detections[0].mask, mask)

    @patch("core.person_detector.YOLO")
    def test_filters_non_person_classes(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mask = np.ones((480, 640), dtype=np.uint8)
        mock_result = self._make_mock_result(
            boxes_xyxy=[[100, 50, 300, 400]],
            masks_data=[mask],
            confs=[0.9],
            cls_ids=[2],  # 2 = car, not person
        )
        mock_model.return_value = [mock_result]

        detector = PersonDetector()
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(detections) == 0

    @patch("core.person_detector.YOLO")
    def test_no_detections_returns_empty(self, mock_yolo_cls):
        mock_model = MagicMock()
        mock_yolo_cls.return_value = mock_model

        mock_result = self._make_mock_result([], None, [], [])
        mock_model.return_value = [mock_result]

        detector = PersonDetector()
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert detections == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_person_detector.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement person_detector.py**

```python
# core/person_detector.py
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
        if result.masks is None or len(result.boxes) == 0:
            return detections

        h, w = frame.shape[:2]

        for i, cls_id in enumerate(result.boxes.cls):
            if int(cls_id) != PERSON_CLASS_ID:
                continue

            bbox_raw = result.boxes.xyxy[i]
            bbox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))
            conf = float(result.boxes.conf[i])

            mask = result.masks.data[i].cpu().numpy().astype(np.uint8)
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            detections.append(PersonDetection(bbox=bbox, mask=mask, confidence=conf))

        return detections
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_person_detector.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/person_detector.py tests/test_person_detector.py
git commit -m "feat: add person detector module with YOLOv8-seg"
```

---

## Task 5: Mosaic Module

**Files:**
- Create: `core/mosaic.py`
- Create: `tests/test_mosaic.py`

- [ ] **Step 1: Write tests for mosaic**

```python
# tests/test_mosaic.py
import numpy as np
import pytest
from core.mosaic import apply_mosaic_to_bbox, apply_mosaic_to_mask


class TestApplyMosaicToBbox:
    def test_pixels_inside_bbox_are_changed(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        bbox = (20, 10, 80, 90)  # x1, y1, x2, y2

        result = apply_mosaic_to_bbox(frame, bbox, block_size=10)

        # Pixels inside bbox should differ from original
        roi_result = result[10:90, 20:80]
        roi_original = original[10:90, 20:80]
        assert not np.array_equal(roi_result, roi_original)

    def test_pixels_outside_bbox_unchanged(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        bbox = (20, 10, 80, 90)

        result = apply_mosaic_to_bbox(frame, bbox, block_size=10)

        # Top strip should be unchanged
        np.testing.assert_array_equal(result[0:10, :], original[0:10, :])

    def test_does_not_modify_input(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        apply_mosaic_to_bbox(frame, (10, 10, 50, 50), block_size=10)
        np.testing.assert_array_equal(frame, original)


class TestApplyMosaicToMask:
    def test_masked_pixels_are_changed(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 1

        result = apply_mosaic_to_mask(frame, mask, block_size=10)

        masked_result = result[mask == 1]
        masked_original = original[mask == 1]
        assert not np.array_equal(masked_result, masked_original)

    def test_unmasked_pixels_unchanged(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1

        result = apply_mosaic_to_mask(frame, mask, block_size=10)

        # Check a region guaranteed outside mask
        np.testing.assert_array_equal(result[0:10, 0:10], original[0:10, 0:10])

    def test_does_not_modify_input(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        original = frame.copy()
        mask = np.ones((100, 100), dtype=np.uint8)
        apply_mosaic_to_mask(frame, mask, block_size=10)
        np.testing.assert_array_equal(frame, original)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_mosaic.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement mosaic.py**

```python
# core/mosaic.py
from typing import Tuple

import cv2
import numpy as np


def apply_mosaic_to_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    block_size: int = 15,
) -> np.ndarray:
    """Apply mosaic (pixelation) blur to a rectangular region.

    Args:
        frame: BGR image (not modified).
        bbox: (x1, y1, x2, y2) region to blur.
        block_size: Mosaic block size in pixels. Larger = more blur.

    Returns:
        New image with mosaic applied to the bbox region.
    """
    result = frame.copy()
    x1, y1, x2, y2 = bbox
    roi = result[y1:y2, x1:x2]

    if roi.size == 0:
        return result

    h, w = roi.shape[:2]
    small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    result[y1:y2, x1:x2] = mosaic

    return result


def apply_mosaic_to_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    block_size: int = 15,
) -> np.ndarray:
    """Apply mosaic blur only to pixels where mask == 1.

    Args:
        frame: BGR image (not modified).
        mask: Binary mask (0 or 1), same H×W as frame.
        block_size: Mosaic block size in pixels.

    Returns:
        New image with mosaic applied only within the mask.
    """
    result = frame.copy()
    h, w = result.shape[:2]

    # Create full-frame mosaic
    small = cv2.resize(result, (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply only where mask is active
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    result[mask_3ch] = mosaic_full[mask_3ch]

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_mosaic.py -v
```

Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/mosaic.py tests/test_mosaic.py
git commit -m "feat: add mosaic blur module with bbox and mask support"
```

---

## Task 6: Pipeline Module

**Files:**
- Create: `core/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write tests for pipeline**

```python
# tests/test_pipeline.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from core.pipeline import Pipeline
from core.person_detector import PersonDetection


class TestPipeline:
    def _make_pipeline(self):
        with patch("core.pipeline.PersonDetector"):
            with patch("core.pipeline.load_encodings", return_value=[np.zeros(128)]):
                return Pipeline(model_path="/fake/model.pkl")

    def test_process_frame_returns_same_shape(self):
        pipeline = self._make_pipeline()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("core.pipeline.find_matching_faces", return_value=[]):
            result = pipeline.process_frame(frame)

        assert result.shape == frame.shape

    def test_no_faces_detected_returns_unmodified(self):
        pipeline = self._make_pipeline()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        with patch("core.pipeline.find_matching_faces", return_value=[]):
            result = pipeline.process_frame(frame)

        np.testing.assert_array_equal(result, frame)

    def test_matched_face_triggers_mosaic(self):
        pipeline = self._make_pipeline()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        original = frame.copy()

        face_loc = (50, 350, 150, 250)  # top, right, bottom, left
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[50:400, 200:400] = 1
        detection = PersonDetection(bbox=(200, 50, 400, 400), mask=mask, confidence=0.9)

        pipeline.detector.detect.return_value = [detection]

        with patch("core.pipeline.find_matching_faces", return_value=[face_loc]):
            result = pipeline.process_frame(frame)

        assert not np.array_equal(result, original)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement pipeline.py**

```python
# core/pipeline.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add core/pipeline.py tests/test_pipeline.py
git commit -m "feat: add pipeline module orchestrating face detection and mosaic"
```

---

## Task 7: GUI — Video Widget

**Files:**
- Create: `gui/video_widget.py`

- [ ] **Step 1: Implement video_widget.py**

```python
# gui/video_widget.py
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

import numpy as np


class VideoWidget(QLabel):
    """Widget that displays video frames, auto-scaling to fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #1a1a2e; border-radius: 8px;")
        self.setText("No input — open an image, video, or start webcam")
        self.setStyleSheet(
            "background-color: #1a1a2e; color: #888; font-size: 14px; border-radius: 8px;"
        )

    def update_frame(self, frame: np.ndarray) -> None:
        """Display a BGR numpy frame, scaled to widget size."""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        rgb = frame[..., ::-1].copy()  # BGR to RGB
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def clear_display(self) -> None:
        """Reset to placeholder text."""
        self.clear()
        self.setText("No input — open an image, video, or start webcam")
```

- [ ] **Step 2: Commit**

```bash
git add gui/video_widget.py
git commit -m "feat: add video display widget"
```

---

## Task 8: GUI — Workers (QThread)

**Files:**
- Create: `gui/workers.py`

- [ ] **Step 1: Implement workers.py**

```python
# gui/workers.py
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import Pipeline


class VideoWorker(QThread):
    """Process video file frame-by-frame in a background thread."""

    frame_ready = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, int)  # current_frame, total_frames
    finished_processing = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_path: str, pipeline: Pipeline, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.pipeline = pipeline
        self._running = True
        self.export_path: str | None = None  # Set before start to export
        self._writer: cv2.VideoWriter | None = None

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self.video_path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.export_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.export_path, fourcc, fps, (w, h))

        idx = 0
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.pipeline.process_frame(frame)
            self.frame_ready.emit(processed)

            if self._writer:
                self._writer.write(processed)

            idx += 1
            self.progress.emit(idx, total)

        cap.release()
        if self._writer:
            self._writer.release()
        self.finished_processing.emit()

    def stop(self):
        self._running = False


class WebcamWorker(QThread):
    """Capture and process webcam frames in a background thread."""

    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.camera_index = camera_index
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera index {self.camera_index}")
            return

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            processed = self.pipeline.process_frame(frame)
            self.frame_ready.emit(processed)

        cap.release()

    def stop(self):
        self._running = False
```

- [ ] **Step 2: Commit**

```bash
git add gui/workers.py
git commit -m "feat: add QThread workers for video and webcam processing"
```

---

## Task 9: GUI — Controls Widget

**Files:**
- Create: `gui/controls_widget.py`

- [ ] **Step 1: Implement controls_widget.py**

```python
# gui/controls_widget.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlsWidget(QWidget):
    """Panel with input buttons, settings sliders, and action buttons."""

    # Signals for main window to connect
    open_image_clicked = pyqtSignal()
    open_video_clicked = pyqtSignal()
    start_webcam_clicked = pyqtSignal()
    save_image_clicked = pyqtSignal()
    export_video_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    # Settings changed signals
    blur_mode_changed = pyqtSignal(str)  # "segmentation" or "bbox"
    mosaic_size_changed = pyqtSignal(int)
    confidence_changed = pyqtSignal(float)
    face_tolerance_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Input Group ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        self.btn_image = QPushButton("📷 Open Image")
        self.btn_video = QPushButton("🎬 Open Video")
        self.btn_webcam = QPushButton("📹 Start Webcam")

        self.btn_image.clicked.connect(self.open_image_clicked.emit)
        self.btn_video.clicked.connect(self.open_video_clicked.emit)
        self.btn_webcam.clicked.connect(self.start_webcam_clicked.emit)

        for btn in [self.btn_image, self.btn_video, self.btn_webcam]:
            btn.setMinimumHeight(36)
            input_layout.addWidget(btn)

        self.status_label = QLabel("Status: Ready")
        input_layout.addWidget(self.status_label)
        layout.addWidget(input_group)

        # --- Settings Group ---
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Blur mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Blur Mode:"))
        self.blur_mode_combo = QComboBox()
        self.blur_mode_combo.addItems(["Segmentation", "Bounding Box"])
        self.blur_mode_combo.currentTextChanged.connect(
            lambda t: self.blur_mode_changed.emit(t.lower().replace(" ", "_"))
        )
        mode_layout.addWidget(self.blur_mode_combo)
        settings_layout.addLayout(mode_layout)

        # Mosaic size slider
        self.mosaic_slider, self.mosaic_label = self._add_slider(
            settings_layout, "Mosaic Size:", 5, 50, 15
        )
        self.mosaic_slider.valueChanged.connect(
            lambda v: (self.mosaic_label.setText(str(v)), self.mosaic_size_changed.emit(v))
        )

        # Confidence slider
        self.conf_slider, self.conf_label = self._add_slider(
            settings_layout, "YOLO Confidence:", 10, 90, 50
        )
        self.conf_slider.valueChanged.connect(
            lambda v: (
                self.conf_label.setText(f"{v / 100:.2f}"),
                self.confidence_changed.emit(v / 100),
            )
        )

        # Face tolerance slider
        self.tol_slider, self.tol_label = self._add_slider(
            settings_layout, "Face Tolerance:", 20, 80, 50
        )
        self.tol_slider.valueChanged.connect(
            lambda v: (
                self.tol_label.setText(f"{v / 100:.2f}"),
                self.face_tolerance_changed.emit(v / 100),
            )
        )

        layout.addWidget(settings_group)

        # --- Actions Group ---
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.btn_save = QPushButton("💾 Save Image")
        self.btn_export = QPushButton("💾 Export Video")
        self.btn_stop = QPushButton("⏹ Stop")

        self.btn_save.clicked.connect(self.save_image_clicked.emit)
        self.btn_export.clicked.connect(self.export_video_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)

        for btn in [self.btn_save, self.btn_export, self.btn_stop]:
            btn.setMinimumHeight(36)
            actions_layout.addWidget(btn)

        layout.addWidget(actions_group)

        # --- Face Model Group ---
        model_group = QGroupBox("Face Model")
        model_layout = QHBoxLayout(model_group)
        self.model_label = QLabel("No model loaded")
        self.btn_reencode = QPushButton("🔄 Re-encode")
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.btn_reencode)
        layout.addWidget(model_group)

        layout.addStretch()

    def _add_slider(self, parent_layout, label_text, min_val, max_val, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(30)
        row.addWidget(slider)
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, value_label

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def set_model_info(self, text: str) -> None:
        self.model_label.setText(text)
```

- [ ] **Step 2: Commit**

```bash
git add gui/controls_widget.py
git commit -m "feat: add controls widget with input, settings and actions panels"
```

---

## Task 10: GUI — Main Window (Integration)

**Files:**
- Create: `gui/main_window.py`

- [ ] **Step 1: Implement main_window.py**

```python
# gui/main_window.py
import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt

from core.face_encoder import encode_faces_from_folder, save_encodings
from core.pipeline import Pipeline
from gui.video_widget import VideoWidget
from gui.controls_widget import ControlsWidget
from gui.workers import VideoWorker, WebcamWorker

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference_faces")
MODELS_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "my_face.pkl")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Mosaic")
        self.setMinimumSize(1000, 600)
        self.resize(1200, 700)

        self.pipeline: Pipeline | None = None
        self.current_frame: np.ndarray | None = None
        self.worker: VideoWorker | WebcamWorker | None = None

        self._setup_ui()
        self._connect_signals()
        self._try_load_model()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.video_widget = VideoWidget()
        self.controls = ControlsWidget()
        self.controls.setFixedWidth(300)

        layout.addWidget(self.video_widget, stretch=1)
        layout.addWidget(self.controls)

    def _connect_signals(self):
        self.controls.open_image_clicked.connect(self._open_image)
        self.controls.open_video_clicked.connect(self._open_video)
        self.controls.start_webcam_clicked.connect(self._start_webcam)
        self.controls.save_image_clicked.connect(self._save_image)
        self.controls.export_video_clicked.connect(self._export_video)
        self.controls.stop_clicked.connect(self._stop_worker)
        self.controls.btn_reencode.clicked.connect(self._reencode_faces)

        self.controls.blur_mode_changed.connect(self._on_blur_mode)
        self.controls.mosaic_size_changed.connect(self._on_mosaic_size)
        self.controls.confidence_changed.connect(self._on_confidence)
        self.controls.face_tolerance_changed.connect(self._on_face_tolerance)

    def _try_load_model(self):
        if os.path.isfile(DEFAULT_MODEL):
            self.pipeline = Pipeline(model_path=DEFAULT_MODEL)
            from core.face_encoder import load_encodings
            count = len(load_encodings(DEFAULT_MODEL))
            self.controls.set_model_info(f"my_face.pkl ✅ ({count} faces)")
            self.controls.set_status("Ready")
        else:
            self.controls.set_model_info("No model — add photos to data/reference_faces/")
            self.controls.set_status("Need face model")

    def _ensure_pipeline(self) -> bool:
        if self.pipeline is None:
            QMessageBox.warning(
                self, "No Face Model",
                "Please add face photos to data/reference_faces/ and click Re-encode.",
            )
            return False
        return True

    # --- Input handlers ---
    def _open_image(self):
        if not self._ensure_pipeline():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self._stop_worker()
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", f"Cannot read image: {path}")
            return
        self.current_frame = self.pipeline.process_frame(frame)
        self.video_widget.update_frame(self.current_frame)
        self.controls.set_status("Image processed")

    def _open_video(self):
        if not self._ensure_pipeline():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self._stop_worker()
        self.worker = VideoWorker(path, self.pipeline)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.progress.connect(
            lambda cur, tot: self.controls.set_status(f"Processing {cur}/{tot}")
        )
        self.worker.finished_processing.connect(
            lambda: self.controls.set_status("Video done")
        )
        self.worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Error", msg)
        )
        self.worker.start()
        self.controls.set_status("Processing video...")

    def _start_webcam(self):
        if not self._ensure_pipeline():
            return
        self._stop_worker()
        self.worker = WebcamWorker(self.pipeline)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Error", msg)
        )
        self.worker.start()
        self.controls.set_status("Webcam active")

    def _on_frame(self, frame: np.ndarray):
        self.current_frame = frame
        self.video_widget.update_frame(frame)

    # --- Actions ---
    def _save_image(self):
        if self.current_frame is None:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", os.path.join(OUTPUT_DIR, "output.png"),
            "Images (*.png *.jpg)"
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self.controls.set_status(f"Saved: {os.path.basename(path)}")

    def _export_video(self):
        if not self._ensure_pipeline():
            return
        src, _ = QFileDialog.getOpenFileName(
            self, "Select Source Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not src:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dst, _ = QFileDialog.getSaveFileName(
            self, "Export To", os.path.join(OUTPUT_DIR, "output.mp4"), "Video (*.mp4)"
        )
        if not dst:
            return
        self._stop_worker()
        self.worker = VideoWorker(src, self.pipeline)
        self.worker.export_path = dst
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.progress.connect(
            lambda cur, tot: self.controls.set_status(f"Exporting {cur}/{tot}")
        )
        self.worker.finished_processing.connect(
            lambda: self.controls.set_status(f"Exported: {os.path.basename(dst)}")
        )
        self.worker.start()
        self.controls.set_status("Exporting video...")

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.controls.set_status("Stopped")

    def _reencode_faces(self):
        if not os.path.isdir(REFERENCE_DIR):
            QMessageBox.warning(self, "Error", f"Folder not found: {REFERENCE_DIR}")
            return
        self.controls.set_status("Encoding faces...")
        self.controls.set_model_info("Encoding...")
        try:
            encodings = encode_faces_from_folder(REFERENCE_DIR)
            if not encodings:
                QMessageBox.warning(self, "No Faces", "No faces found in reference folder.")
                self.controls.set_model_info("No faces found")
                return
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_encodings(encodings, DEFAULT_MODEL)
            self.pipeline = Pipeline(model_path=DEFAULT_MODEL)
            self.controls.set_model_info(f"my_face.pkl ✅ ({len(encodings)} faces)")
            self.controls.set_status("Ready")
        except Exception as e:
            QMessageBox.critical(self, "Encoding Error", str(e))
            self.controls.set_status("Encoding failed")

    # --- Settings handlers ---
    def _on_blur_mode(self, mode: str):
        if self.pipeline:
            self.pipeline.use_segmentation = mode == "segmentation"

    def _on_mosaic_size(self, size: int):
        if self.pipeline:
            self.pipeline.mosaic_block_size = size

    def _on_confidence(self, conf: float):
        if self.pipeline:
            self.pipeline.yolo_confidence = conf

    def _on_face_tolerance(self, tol: float):
        if self.pipeline:
            self.pipeline.face_tolerance = tol

    def closeEvent(self, event):
        self._stop_worker()
        super().closeEvent(event)
```

- [ ] **Step 2: Commit**

```bash
git add gui/main_window.py
git commit -m "feat: add main window integrating all GUI components"
```

---

## Task 11: End-to-End Integration Test

- [ ] **Step 1: Run all unit tests**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python -m pytest tests/ -v
```

Expected: All tests PASS

- [ ] **Step 2: Manual smoke test — encode faces**

Put at least 5-10 photos of your face into `data/reference_faces/`. Then run:

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python -c "
from core.face_encoder import encode_faces_from_folder, save_encodings
import os
encodings = encode_faces_from_folder('data/reference_faces')
print(f'Encoded {len(encodings)} faces')
os.makedirs('data/models', exist_ok=True)
save_encodings(encodings, 'data/models/my_face.pkl')
print('Saved to data/models/my_face.pkl')
"
```

Expected: `Encoded N faces` where N > 0

- [ ] **Step 3: Launch the GUI**

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python main.py
```

Expected: PyQt6 window opens. Face model shows loaded. Test:
1. Open an image with your face → your body should be mosaic'd
2. Open an image with other people → they should NOT be mosaic'd
3. Start webcam → your body gets mosaic'd in realtime
4. Adjust sliders → mosaic intensity and detection sensitivity change

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete face mosaic app with GUI, pipeline, and tests"
```

---

## Verification Plan

### Automated Tests

```bash
cd /Volumes/Data/Code/AI/face-mosaic
python -m pytest tests/ -v --tb=short
```

All 17+ tests should pass covering: face encoder (6 tests), face matcher (5 tests), person detector (3 tests), mosaic (6 tests), pipeline (3 tests).

### Manual Verification

1. **Encode test**: Drop 5+ face photos into `data/reference_faces/`, click Re-encode, verify model loads
2. **Image test**: Open a group photo containing you → only you are mosaic'd
3. **Video test**: Open a video → export with mosaic applied
4. **Webcam test**: Start webcam, verify ~15-30 FPS on M2, your body is mosaic'd
5. **Settings test**: Toggle between Segmentation/Bounding Box, adjust sliders, verify changes take effect
6. **Edge cases**: Open image with no people, image with only strangers (no blur expected)
