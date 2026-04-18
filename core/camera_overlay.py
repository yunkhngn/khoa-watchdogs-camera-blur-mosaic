import time
import cv2
import numpy as np

def apply_cctv_overlay(frame: np.ndarray) -> np.ndarray:
    """Applies a CCTV / Security Camera style overlay to the entire frame."""
    # We do not use frame.copy() immediately if we can avoid it for speed, 
    # but modifying in-place is usually fine for OpenCV. 
    # Since we want to be safe pipeline-wise:
    result = frame.copy()
    h, w = result.shape[:2]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_white = (255, 255, 255)
    color_red = (0, 0, 255)
    
    # 1. Bule/Greenish tint (Desaturate and tint to look like old CCTV)
    # Convert to HSV, reduce saturation, back to BGR
    # But a faster way is just blending with a dark bluish-green color
    # Let's skip complex tinting to maintain 30FPS, and just do scanlines & UI.
    
    # 2. Scanlines over the whole image (every 3rd row is darkened by 10%)
    result[::3, :] = (result[::3, :] * 0.9).astype(np.uint8)
    
    # 3. Add vignette (darkened corners)
    # Create a simple static vignette mask if needed, but for FPS sake, we can skip Gaussian blur
    # Let's just draw some UI overlays.

    padding = 25
    
    # REC UI (Top Right)
    if int(time.time() * 2) % 2 == 0:  # Blinking effect (2 times per sec)
        cv2.circle(result, (w - padding - 70, padding + 10), 6, color_red, cv2.FILLED)
    cv2.putText(result, "REC", (w - padding - 55, padding + 15), font, 0.6, color_white, 2, cv2.LINE_AA)
    
    cv2.putText(result, "SPY_BETA_V1.2", (w - padding - 130, padding + 35), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Camera Name (Top Left)
    cv2.putText(result, "CAM 01 - ROOT SECTOR", (padding, padding + 15), font, 0.5, color_white, 1, cv2.LINE_AA)
    
    # Timestamp (Bottom Left)
    timestamp = time.strftime("%Y/%m/%d %H:%M:%S")
    time_ms = f"{int(time.time() * 1000) % 1000:03d}" # milliseconds simulating tape counter
    
    full_time_str = f"{timestamp} : {time_ms}"
    cv2.putText(result, full_time_str, (padding, h - padding), font, 0.6, color_white, 1, cv2.LINE_AA)
    
    # System Status (Bottom Right)
    cv2.putText(result, "NO SIGNAL LOSS", (w - padding - 150, h - padding), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return result
