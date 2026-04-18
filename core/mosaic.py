import random
from typing import Tuple

import cv2
import numpy as np


def _create_glitch_mosaic(img: np.ndarray) -> np.ndarray:
    """Takes an image and applies chromatic aberration and horizontal slice shifts."""
    glitched = img.copy()
    h, w = glitched.shape[:2]
    
    # Random frequency so it doesn't glitch constantly like crazy
    if random.random() > 0.7:
        return glitched

    # 1. Chromatic Aberration (Shift Red and Blue channels)
    shift_x = random.randint(3, 8)
    if random.random() > 0.5: 
        shift_x = -shift_x
    
    # BGR format: 0=B, 1=G, 2=R
    temp_R = np.roll(glitched[:, :, 2], shift_x, axis=1)
    temp_B = np.roll(glitched[:, :, 0], -shift_x, axis=1)
    
    glitched[:, :, 2] = temp_R
    glitched[:, :, 0] = temp_B
    
    # 2. Horizontal slices displacement
    num_slices = random.randint(1, 6)
    for _ in range(num_slices):
        slice_y = random.randint(0, max(1, h - 20))
        slice_h = random.randint(5, 30)
        slice_shift = random.randint(-20, 20)
        glitched[slice_y:slice_y+slice_h] = np.roll(
            glitched[slice_y:slice_y+slice_h], slice_shift, axis=1
        )
        
    # 3. Add faint scanline darkening
    glitched[::2, :] = (glitched[::2, :] * 0.85).astype(np.uint8)
    
    return glitched


def draw_hacker_box(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Draw a cyberpunk-style targeting box with [ UNKNOWN ] text in-place."""
    x1, y1, x2, y2 = bbox
    color_neon = (0, 255, 0) # Green for hacker style
    color_red = (0, 0, 255)
    
    # Draw corners
    length = int((x2 - x1) * 0.15)
    thickness = 2
    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color_neon, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color_neon, thickness)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color_neon, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color_neon, thickness)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color_neon, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color_neon, thickness)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color_neon, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color_neon, thickness)
    
    # Add a thin full bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 100, 0), 1)

    # Text backdrop
    text = f"TARGET: UNKNOWN [ID:{random.randint(1000,9999)}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    
    # Glitch the text position slightly sometimes
    txt_y = max(y1 - 10, th + 10)
    if random.random() > 0.85:
        txt_y += random.randint(-4, 4)
        
    cv2.rectangle(frame, (x1, txt_y - th - 5), (x1 + tw + 10, txt_y + 5), (0, 40, 0), cv2.FILLED)
    cv2.putText(frame, text, (x1 + 5, txt_y), font, font_scale, color_neon, 1, cv2.LINE_AA)
    
    # Add a small red recording indicator sometimes blinking
    import time
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (x1 + tw + 20, txt_y - th // 2), 4, color_red, cv2.FILLED)


def draw_human_box(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
    """Draw a basic gray bounding box indicating a 'HUMAN'."""
    x1, y1, x2, y2 = bbox
    color_gray = (170, 170, 170)
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_gray, 2)

    text = "HUMAN"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
    
    txt_y = max(y1 - 10, th + 10)
    cv2.rectangle(frame, (x1, txt_y - th - 5), (x1 + tw + 10, txt_y + 5), (50, 50, 50), cv2.FILLED)
    cv2.putText(frame, text, (x1 + 5, txt_y), font, font_scale, color_gray, 1, cv2.LINE_AA)

def apply_mosaic_to_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    block_size: int = 15,
) -> np.ndarray:
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
    
    # Apply glitch
    mosaic = _create_glitch_mosaic(mosaic)
    
    result[y1:y2, x1:x2] = mosaic
    return result


def apply_mosaic_to_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    block_size: int = 15,
) -> np.ndarray:
    result = frame.copy()
    h, w = result.shape[:2]

    small = cv2.resize(
        result,
        (max(1, w // block_size), max(1, h // block_size)),
        interpolation=cv2.INTER_LINEAR,
    )
    mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Apply glitch
    mosaic_full = _create_glitch_mosaic(mosaic_full)

    mask_3ch = np.stack([mask] * 3, axis=-1).astype(bool)
    result[mask_3ch] = mosaic_full[mask_3ch]

    return result
