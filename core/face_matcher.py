from typing import List, Tuple

import numpy as np
import face_recognition

FaceLocation = Tuple[int, int, int, int]  # (top, right, bottom, left)


def find_matching_faces(
    frame: np.ndarray,
    known_encodings: List[np.ndarray],
    tolerance: float = 0.6,
    model: str = "hog",
) -> List[FaceLocation]:
    """Detect faces in frame, return locations of those matching known encodings.

    Args:
        frame: BGR or RGB image as numpy array.
        known_encodings: List of 128-d face encoding vectors.
        tolerance: Distance threshold. Lower = stricter matching.
        model: 'hog' (faster, CPU) or 'cnn' (more accurate, GPU).

    Returns:
        List of (top, right, bottom, left) tuples for matched faces.
    """
    if not known_encodings:
        return []

    face_locations = face_recognition.face_locations(frame, model=model)
    if not face_locations:
        return []

    face_encodings = face_recognition.face_encodings(frame, face_locations)

    matched_locations = []
    for location, encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(
            known_encodings, encoding, tolerance=tolerance
        )
        if any(matches):
            matched_locations.append(location)

    return matched_locations
