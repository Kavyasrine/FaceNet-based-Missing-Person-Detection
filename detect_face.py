import cv2
from mtcnn import MTCNN
import numpy as np

_mtcnn_detector = None

def load_mtcnn():
    """
    Load and return a global MTCNN detector.
    """
    global _mtcnn_detector
    if _mtcnn_detector is None:
        _mtcnn_detector = MTCNN()
        print("MTCNN detector loaded successfully.")
    return _mtcnn_detector

def detect_face(img, detector, threshold=[0.6, 0.7, 0.7]):
    """
    Detect faces in an image using MTCNN with configurable thresholds.
    Returns a list of boxes [x, y, w, h] for detected faces above confidence.
    - threshold: List of [stage1, stage2, stage3] probabilities (default matches data_preprocess.py).
    """
    if detector is None:
        detector = load_mtcnn()

    # Use MTCNN's detect_faces with thresholds
    results = detector.detect_faces(img)
    boxes = []
    for r in results:
        # Check if confidence meets the threshold (use the final stage confidence)
        if r['confidence'] >= threshold[2]:  # Stage 3 threshold for final detection
            x, y, w, h = r['box']
            x, y = max(0, x), max(0, y)  # Ensure non-negative
            boxes.append([x, y, w, h])
    return boxes