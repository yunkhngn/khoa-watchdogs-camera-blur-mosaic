import time
import cv2
import numpy as np

def apply_cctv_overlay(frame: np.ndarray) -> np.ndarray:
    """Applies a realistic CCTV Security Camera style overlay matching the reference image."""
    result = frame.copy()
    h, w = result.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_white = (230, 230, 230)
    color_red = (60, 60, 255) # BGR
    
    padding = 40
    corner_len = 30
    thickness = 2
    
    # --- 1. Draw Viewport Corners ---
    # Top-Left
    cv2.line(result, (padding, padding), (padding + corner_len, padding), color_white, thickness)
    cv2.line(result, (padding, padding), (padding, padding + corner_len), color_white, thickness)
    # Top-Right
    cv2.line(result, (w - padding, padding), (w - padding - corner_len, padding), color_white, thickness)
    cv2.line(result, (w - padding, padding), (w - padding, padding + corner_len), color_white, thickness)
    # Bottom-Left
    cv2.line(result, (padding, h - padding), (padding + corner_len, h - padding), color_white, thickness)
    cv2.line(result, (padding, h - padding), (padding, h - padding - corner_len), color_white, thickness)
    # Bottom-Right
    cv2.line(result, (w - padding, h - padding), (w - padding - corner_len, h - padding), color_white, thickness)
    cv2.line(result, (w - padding, h - padding), (w - padding, h - padding - corner_len), color_white, thickness)

    # --- 2. Top Left: REC ---
    # Blinking red dot
    if int(time.time() * 2) % 2 == 0:
        cv2.circle(result, (padding + 25, padding + 25), 12, color_red, cv2.FILLED)
    cv2.putText(result, "REC", (padding + 48, padding + 34), font, 1.0, color_white, 2, cv2.LINE_AA)
    
    # --- 3. Top Right: CAM ---
    cv2.putText(result, "CAM4", (w - padding - 85, padding + 34), font, 0.8, color_white, 2, cv2.LINE_AA)
    
    # --- 4. Bottom Left: Camera Specs ---
    specs_text = "ISO 800    F 2.4    HD1080P    AWB"
    cv2.putText(result, specs_text, (padding + 20, h - padding - 15), font, 0.6, color_white, 1, cv2.LINE_AA)
    
    # --- 5. Bottom Right: Audio Channels ---
    # Draw CH1, CH2 labels
    ch_x = w - padding - 220
    ch_y = h - padding - 30
    cv2.putText(result, "CH1", (ch_x, ch_y), font, 0.4, color_white, 1, cv2.LINE_AA)
    cv2.putText(result, "CH2", (ch_x, ch_y + 15), font, 0.4, color_white, 1, cv2.LINE_AA)
    
    # Draw volume blocks
    block_w, block_h = 8, 8
    gap = 4
    for i in range(12):
        bx = ch_x + 35 + i * (block_w + gap)
        # CH1
        if i < 8: # Arbitrary level
            cv2.rectangle(result, (bx, ch_y - 8), (bx + block_w, ch_y - 8 + block_h), color_white, cv2.FILLED)
        else:
            cv2.rectangle(result, (bx, ch_y - 8), (bx + block_w, ch_y - 8 + block_h), color_white, 1)
        
        # CH2
        if i < 7:
            cv2.rectangle(result, (bx, ch_y + 7), (bx + block_w, ch_y + 7 + block_h), color_white, cv2.FILLED)
        else:
            cv2.rectangle(result, (bx, ch_y + 7), (bx + block_w, ch_y + 7 + block_h), color_white, 1)

    # --- 6. Right Edge: Exposure Scale (2 to -2) ---
    scale_x = w - padding + 5
    scale_y_center = h // 2
    step = 40
    labels = ["2", "1", "0", "-1", "-2"]
    
    # Draw main line
    cv2.line(result, (scale_x + 20, scale_y_center - 2*step), (scale_x + 20, scale_y_center + 2*step), color_white, 1)
    
    for i, lbl in enumerate(labels):
        y_pos = scale_y_center - 2*step + i*step
        # Draw tick
        cv2.line(result, (scale_x + 15, y_pos), (scale_x + 25, y_pos), color_white, 1)
        # Draw small sub-ticks
        if i < 4:
            for j in range(1, 4):
                sub_y = y_pos + j * (step // 4)
                cv2.line(result, (scale_x + 18, sub_y), (scale_x + 20, sub_y), color_white, 1)
        # Draw text next to major ticks
        cv2.putText(result, lbl, (scale_x, y_pos + 4), font, 0.4, color_white, 1, cv2.LINE_AA)

    return result
