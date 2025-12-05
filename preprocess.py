import cv2
import numpy as np
from facenet import prewhiten  # Import for consistent preprocessing

def align_single_face(img, box, output_size=160):
    """
    Crop and resize a single face from image, with consistent FaceNet preprocessing.
    box = [x, y, w, h]
    Returns prewhitened face array ready for embedding.
    """
    x, y, w, h = box
    h_img, w_img = img.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)

    if x2 <= x1 or y2 <= y1:
        return None

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None

    face = cv2.resize(face, (output_size, output_size))
    # Apply consistent normalization (prewhiten, same as training/inference)
    face = prewhiten(face.astype(np.float32))
    return face