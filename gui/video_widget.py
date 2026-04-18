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
