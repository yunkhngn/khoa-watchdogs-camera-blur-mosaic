import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.pipeline import Pipeline


class VideoWorker(QThread):
    """Process video file frame-by-frame in a background thread."""

    frame_ready = pyqtSignal(np.ndarray)
    progress = pyqtSignal(int, int)  # current_frame, total_frames
    finished_processing = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, video_path: str, pipeline: Pipeline, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.pipeline = pipeline
        self._running = True
        self.export_path: str | None = None  # Set before start to export
        self._writer: cv2.VideoWriter | None = None

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.error.emit(f"Cannot open video: {self.video_path}")
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.export_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(self.export_path, fourcc, fps, (w, h))

        idx = 0
        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed = self.pipeline.process_frame(frame)
            self.frame_ready.emit(processed)

            if self._writer:
                self._writer.write(processed)

            idx += 1
            self.progress.emit(idx, total)

        cap.release()
        if self._writer:
            self._writer.release()
        self.finished_processing.emit()

    def stop(self):
        self._running = False


class WebcamWorker(QThread):
    """Capture and process webcam frames in a background thread."""

    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, pipeline: Pipeline, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.camera_index = camera_index
        self._running = True

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera index {self.camera_index}")
            return

        while self._running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
            processed = self.pipeline.process_frame(frame)
            self.frame_ready.emit(processed)

        cap.release()

    def stop(self):
        self._running = False
