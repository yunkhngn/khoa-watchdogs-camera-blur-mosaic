import os
import pickle
from typing import List

import numpy as np
import face_recognition

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def encode_faces_from_folder(folder_path: str) -> List[np.ndarray]:
    """Scan folder for images, detect faces, return list of 128-d encodings."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    encodings = []
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            continue

        filepath = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(filepath)
        face_encs = face_recognition.face_encodings(image)

        if face_encs:
            # Take the first (largest) face from each image
            encodings.append(face_encs[0])

    return encodings


def save_encodings(encodings: List[np.ndarray], output_path: str) -> None:
    """Save encodings list to a pickle file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(encodings, f)


def load_encodings(model_path: str) -> List[np.ndarray]:
    """Load encodings list from a pickle file."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)
