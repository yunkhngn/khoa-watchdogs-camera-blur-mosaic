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
