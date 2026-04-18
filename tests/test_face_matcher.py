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
            mock_fr.compare_faces.side_effect = [[True], [False]]

            result = find_matching_faces(frame, self.known, tolerance=0.6)
            assert result == [loc1]
