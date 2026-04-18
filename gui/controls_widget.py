from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QComboBox,
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlsWidget(QWidget):
    """Panel with input buttons, settings sliders, and action buttons."""

    # Signals for main window to connect
    open_image_clicked = pyqtSignal()
    open_video_clicked = pyqtSignal()
    start_webcam_clicked = pyqtSignal()
    save_image_clicked = pyqtSignal()
    export_video_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()

    # Settings changed signals
    blur_mode_changed = pyqtSignal(str)  # "segmentation" or "bbox"
    mosaic_size_changed = pyqtSignal(int)
    confidence_changed = pyqtSignal(float)
    face_tolerance_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Input Group ---
        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        self.btn_image = QPushButton("Open Image")
        self.btn_video = QPushButton("Open Video")
        self.btn_webcam = QPushButton("Start Webcam")

        self.btn_image.clicked.connect(self.open_image_clicked.emit)
        self.btn_video.clicked.connect(self.open_video_clicked.emit)
        self.btn_webcam.clicked.connect(self.start_webcam_clicked.emit)

        for btn in [self.btn_image, self.btn_video, self.btn_webcam]:
            btn.setMinimumHeight(36)
            input_layout.addWidget(btn)

        self.status_label = QLabel("Status: Ready")
        input_layout.addWidget(self.status_label)
        layout.addWidget(input_group)

        # --- Settings Group ---
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Blur mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Blur Mode:"))
        self.blur_mode_combo = QComboBox()
        self.blur_mode_combo.addItems(["Segmentation", "Bounding Box"])
        self.blur_mode_combo.currentTextChanged.connect(
            lambda t: self.blur_mode_changed.emit(t.lower().replace(" ", "_"))
        )
        mode_layout.addWidget(self.blur_mode_combo)
        settings_layout.addLayout(mode_layout)

        # Mosaic size slider
        self.mosaic_slider, self.mosaic_label = self._add_slider(
            settings_layout, "Mosaic Size:", 5, 50, 15
        )
        self.mosaic_slider.valueChanged.connect(
            lambda v: (self.mosaic_label.setText(str(v)), self.mosaic_size_changed.emit(v))
        )

        # Confidence slider
        self.conf_slider, self.conf_label = self._add_slider(
            settings_layout, "YOLO Confidence:", 10, 90, 50
        )
        self.conf_slider.valueChanged.connect(
            lambda v: (
                self.conf_label.setText(f"{v / 100:.2f}"),
                self.confidence_changed.emit(v / 100),
            )
        )

        # Face tolerance slider
        self.tol_slider, self.tol_label = self._add_slider(
            settings_layout, "Face Tolerance:", 20, 80, 50
        )
        self.tol_slider.valueChanged.connect(
            lambda v: (
                self.tol_label.setText(f"{v / 100:.2f}"),
                self.face_tolerance_changed.emit(v / 100),
            )
        )

        layout.addWidget(settings_group)

        # --- Actions Group ---
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)

        self.btn_save = QPushButton("Save Image")
        self.btn_export = QPushButton("Export Video")
        self.btn_stop = QPushButton("Stop")

        self.btn_save.clicked.connect(self.save_image_clicked.emit)
        self.btn_export.clicked.connect(self.export_video_clicked.emit)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)

        for btn in [self.btn_save, self.btn_export, self.btn_stop]:
            btn.setMinimumHeight(36)
            actions_layout.addWidget(btn)

        layout.addWidget(actions_group)

        # --- Face Model Group ---
        model_group = QGroupBox("Face Model")
        model_layout = QHBoxLayout(model_group)
        self.model_label = QLabel("No model loaded")
        self.btn_reencode = QPushButton("Re-encode Face Model")
        self.btn_reencode.setObjectName("primaryBtn")
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.btn_reencode)
        layout.addWidget(model_group)

        layout.addStretch()

    def _add_slider(self, parent_layout, label_text, min_val, max_val, default):
        row = QHBoxLayout()
        row.addWidget(QLabel(label_text))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        value_label = QLabel(str(default))
        value_label.setMinimumWidth(30)
        row.addWidget(slider)
        row.addWidget(value_label)
        parent_layout.addLayout(row)
        return slider, value_label

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def set_model_info(self, text: str) -> None:
        self.model_label.setText(text)
