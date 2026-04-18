# Hacker CCTV Face Mosaic

An intelligent face recognition project with a high-end Cyberpunk Hacker aesthetic. This application automatically detects you, identifies your entire body, and applies a mosaic blur with glitch effects, while leaving other people completely unaffected.

## Key Features

- **Selective Identity Blurring:** Blurs ONLY the face and body registered in the reference_faces directory.
- **Instance Segmentation:** Uses YOLOv8-seg to create masks that follow your body contours precisely, rather than just using rough bounding boxes.
- **Glitch & CCTV Aesthetics:** 
  - Modern surveillance interface with scanlines, REC indicator, exposure scales, and audio bars.
  - Grayscale mode for a realistic security camera look.
  - Chromatic aberration glitch effects and UNKNOWN target panels.
- **M1/M2 Optimization:** Achieves 30+ FPS on Apple Silicon through face detection caching and multi-threaded processing via PyQt6.

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd face-mosaic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add face data:
   - Create the directory: data/reference_faces/
   - Place 5-10 clear photos of your face in that directory.

## Usage

1. Launch the application:
   ```bash
   python3 main.py
   ```

2. Train the Face Model:
   - Click the "Re-encode Face Model" button in the bottom right. Wait for the status to show success (e.g., "19 faces").

3. Operation Modes:
   - Start Webcam: Real-time monitoring and blurring.
   - Open Image/Video: Process existing local files.
   - Export Video: Automatically process and save videos to the output directory.

## Technical Stack

- **AI Core:** Built with face_recognition (dlib) and ultralytics (YOLOv8-seg).
- **GUI:** Premium Dark Mode interface powered by PyQt6.
- **Processing:** OpenCV, NumPy, and Pillow.

## Settings and Configuration

- **Blur Mode:** Toggle between Segmentation (high accuracy) and Bounding Box (high speed).
- **Mosaic Size:** Adjust the pixelation intensity.
- **Face Tolerance:** Adjust recognition sensitivity (lower is stricter).
- **Grayscale Filter:** Toggle between color and black-and-white camera views.
