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
