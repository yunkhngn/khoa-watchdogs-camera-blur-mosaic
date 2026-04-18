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
