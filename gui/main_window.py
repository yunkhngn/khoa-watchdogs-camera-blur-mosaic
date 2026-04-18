import os
import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QFileDialog, QMessageBox,
)
from PyQt6.QtCore import Qt

from core.face_encoder import encode_faces_from_folder, save_encodings
from core.pipeline import Pipeline
from gui.video_widget import VideoWidget
from gui.controls_widget import ControlsWidget
from gui.workers import VideoWorker, WebcamWorker

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
REFERENCE_DIR = os.path.join(DATA_DIR, "reference_faces")
MODELS_DIR = os.path.join(DATA_DIR, "models")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
DEFAULT_MODEL = os.path.join(MODELS_DIR, "my_face.pkl")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Mosaic")
        self.setMinimumSize(1000, 600)
        self.resize(1200, 700)

        self.pipeline: Pipeline | None = None
        self.current_frame: np.ndarray | None = None
        self.worker: VideoWorker | WebcamWorker | None = None

        self._setup_ui()
        self._connect_signals()
        self._try_load_model()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.video_widget = VideoWidget()
        self.controls = ControlsWidget()
        self.controls.setFixedWidth(300)

        layout.addWidget(self.video_widget, stretch=1)
        layout.addWidget(self.controls)

    def _connect_signals(self):
        self.controls.open_image_clicked.connect(self._open_image)
        self.controls.open_video_clicked.connect(self._open_video)
        self.controls.start_webcam_clicked.connect(self._start_webcam)
        self.controls.save_image_clicked.connect(self._save_image)
        self.controls.export_video_clicked.connect(self._export_video)
        self.controls.stop_clicked.connect(self._stop_worker)
        self.controls.btn_reencode.clicked.connect(self._reencode_faces)

        self.controls.blur_mode_changed.connect(self._on_blur_mode)
        self.controls.mosaic_size_changed.connect(self._on_mosaic_size)
        self.controls.confidence_changed.connect(self._on_confidence)
        self.controls.face_tolerance_changed.connect(self._on_face_tolerance)

    def _try_load_model(self):
        if os.path.isfile(DEFAULT_MODEL):
            self.pipeline = Pipeline(model_path=DEFAULT_MODEL)
            from core.face_encoder import load_encodings
            count = len(load_encodings(DEFAULT_MODEL))
            self.controls.set_model_info(f"my_face.pkl ✅ ({count} faces)")
            self.controls.set_status("Ready")
        else:
            self.controls.set_model_info("No model — add photos to data/reference_faces/")
            self.controls.set_status("Need face model")

    def _ensure_pipeline(self) -> bool:
        if self.pipeline is None:
            QMessageBox.warning(
                self, "No Face Model",
                "Please add face photos to data/reference_faces/ and click Re-encode.",
            )
            return False
        return True

    # --- Input handlers ---
    def _open_image(self):
        if not self._ensure_pipeline():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if not path:
            return
        self._stop_worker()
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, "Error", f"Cannot read image: {path}")
            return
        self.current_frame = self.pipeline.process_frame(frame)
        self.video_widget.update_frame(self.current_frame)
        self.controls.set_status("Image processed")

    def _open_video(self):
        if not self._ensure_pipeline():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        self._stop_worker()
        self.worker = VideoWorker(path, self.pipeline)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.progress.connect(
            lambda cur, tot: self.controls.set_status(f"Processing {cur}/{tot}")
        )
        self.worker.finished_processing.connect(
            lambda: self.controls.set_status("Video done")
        )
        self.worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Error", msg)
        )
        self.worker.start()
        self.controls.set_status("Processing video...")

    def _start_webcam(self):
        if not self._ensure_pipeline():
            return
        self._stop_worker()
        self.worker = WebcamWorker(self.pipeline)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.error.connect(
            lambda msg: QMessageBox.warning(self, "Error", msg)
        )
        self.worker.start()
        self.controls.set_status("Webcam active")

    def _on_frame(self, frame: np.ndarray):
        self.current_frame = frame
        self.video_widget.update_frame(frame)

    # --- Actions ---
    def _save_image(self):
        if self.current_frame is None:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", os.path.join(OUTPUT_DIR, "output.png"),
            "Images (*.png *.jpg)"
        )
        if path:
            cv2.imwrite(path, self.current_frame)
            self.controls.set_status(f"Saved: {os.path.basename(path)}")

    def _export_video(self):
        if not self._ensure_pipeline():
            return
        src, _ = QFileDialog.getOpenFileName(
            self, "Select Source Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)"
        )
        if not src:
            return
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dst, _ = QFileDialog.getSaveFileName(
            self, "Export To", os.path.join(OUTPUT_DIR, "output.mp4"), "Video (*.mp4)"
        )
        if not dst:
            return
        self._stop_worker()
        self.worker = VideoWorker(src, self.pipeline)
        self.worker.export_path = dst
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.progress.connect(
            lambda cur, tot: self.controls.set_status(f"Exporting {cur}/{tot}")
        )
        self.worker.finished_processing.connect(
            lambda: self.controls.set_status(f"Exported: {os.path.basename(dst)}")
        )
        self.worker.start()
        self.controls.set_status("Exporting video...")

    def _stop_worker(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            self.worker = None
        self.controls.set_status("Stopped")

    def _reencode_faces(self):
        if not os.path.isdir(REFERENCE_DIR):
            QMessageBox.warning(self, "Error", f"Folder not found: {REFERENCE_DIR}")
            return
        self.controls.set_status("Encoding faces...")
        self.controls.set_model_info("Encoding...")
        try:
            encodings = encode_faces_from_folder(REFERENCE_DIR)
            if not encodings:
                QMessageBox.warning(self, "No Faces", "No faces found in reference folder.")
                self.controls.set_model_info("No faces found")
                return
            os.makedirs(MODELS_DIR, exist_ok=True)
            save_encodings(encodings, DEFAULT_MODEL)
            self.pipeline = Pipeline(model_path=DEFAULT_MODEL)
            self.controls.set_model_info(f"my_face.pkl ✅ ({len(encodings)} faces)")
            self.controls.set_status("Ready")
        except Exception as e:
            QMessageBox.critical(self, "Encoding Error", str(e))
            self.controls.set_status("Encoding failed")

    # --- Settings handlers ---
    def _on_blur_mode(self, mode: str):
        if self.pipeline:
            self.pipeline.use_segmentation = mode == "segmentation"

    def _on_mosaic_size(self, size: int):
        if self.pipeline:
            self.pipeline.mosaic_block_size = size

    def _on_confidence(self, conf: float):
        if self.pipeline:
            self.pipeline.yolo_confidence = conf

    def _on_face_tolerance(self, tol: float):
        if self.pipeline:
            self.pipeline.face_tolerance = tol

    def closeEvent(self, event):
        self._stop_worker()
        super().closeEvent(event)
