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
    small = cv2.resize(
        roi,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
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
    small = cv2.resize(
        result,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Apply only where mask is active
    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    result[mask_3ch] = mosaic_full[mask_3ch]

    return result
