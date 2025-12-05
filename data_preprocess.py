import os
import cv2
import numpy as np
from detect_face import load_mtcnn, detect_face
from imageio.v2 import imread
import facenet  # Import for consistent preprocessing

def align_face(img, boxes, output_size=160):
    """Align and crop the first detected face, with consistent preprocessing."""
    if boxes is None or len(boxes) == 0:
        return None

    bb = np.array(boxes[0]).astype(int)
    x1, y1, x2, y2 = bb[:4]
    h, w = img.shape[:2]

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # Resize to FaceNet input size
    face = cv2.resize(face, (output_size, output_size))
    # Convert to RGB if needed (imageio loads as RGB, but ensure)
    if face.shape[2] == 3:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # If loaded as BGR, convert
    # Apply consistent normalization (same as training/inference)
    face = facenet.prewhiten(face.astype(np.float32))
    return face

def preprocess_image(img_path, detector=None, threshold=[0.6, 0.7, 0.7]):
    """Load image, detect face, return aligned and prewhitened face."""
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return None

    if detector is None:
        detector = load_mtcnn()

    try:
        img = imread(img_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        boxes = detect_face(img, detector, threshold)
        aligned = align_face(img, boxes)
        return aligned

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def process_dataset(dataset_dir=r"C:\Users\rkavy\Downloads\PersonMissed\Full project_face recog\train_img",
                    output_dir=r"C:\Users\rkavy\Downloads\PersonMissed\Full project_face recog\pre_img"):
    """Preprocess all images in dataset and save aligned, prewhitened faces."""

    detector = load_mtcnn()
    os.makedirs(output_dir, exist_ok=True)

    for person_name in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        out_person_dir = os.path.join(output_dir, person_name)
        os.makedirs(out_person_dir, exist_ok=True)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            aligned_face = preprocess_image(img_path, detector)
            if aligned_face is not None:
                # Save as float32 numpy array (prewhitened, so training can load directly)
                save_path = os.path.join(out_person_dir, img_name.replace('.jpg', '.npy').replace('.png', '.npy'))
                np.save(save_path, aligned_face)
                print(f"Saved: {save_path}")
            else:
                print(f"Skipped (no face found): {img_name}")

if __name__ == "__main__":
    process_dataset()
    print("Preprocessing completed successfully.")