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
