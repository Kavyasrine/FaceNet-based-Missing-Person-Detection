import cv2
import numpy as np
import pickle
import os
import tensorflow as tf
from detect_face import load_mtcnn, detect_face
from facenet import Facenet, prewhiten  # Import prewhiten for consistency

# ---------------- SETTINGS ----------------
VIDEO_PATH = input("Enter CCTV video filename (example: cctv_1.mp4): ").strip()
CLASSIFIER_PATH = r"C:\Users\rkavy\Downloads\PersonMissed\class\classifier.pkl"
MODEL_PATH = r"C:\Users\rkavy\Downloads\PersonMissed\model\20170511-185253.pb"
CONFIDENCE_THRESHOLD = 0.5  # Adjustable threshold for predictions (0.0-1.0)

# Disable TF eager for .pb model
tf.compat.v1.disable_eager_execution()

# ---------------- LOAD FACENET ----------------
print("Loading FaceNet model...")
facenet_model = Facenet(model_path=MODEL_PATH)
sess = facenet_model.load_model()

# ---------------- LOAD CLASSIFIER ----------------
if not os.path.exists(CLASSIFIER_PATH):
    raise FileNotFoundError("Run train_main.py first to create classifier.pkl")

with open(CLASSIFIER_PATH, "rb") as f:
    data = pickle.load(f)
model = data["model"]
label_encoder = data["le"]

# ---------------- LOAD MTCNN ----------------
print("Loading MTCNN detector...")
detector = load_mtcnn()

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"{VIDEO_PATH} not found or cannot be opened.")

frame_interval = 3
frame_count = 0
print("Start Recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = detect_face(rgb, detector)

    if boxes:
        for box in boxes:
            # Align face (crop and resize)
            x, y, w, h = box
            h_img, w_img = rgb.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w_img, x + w), min(h_img, y + h)

            if x2 <= x1 or y2 <= y1:
                continue

            face = rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face = cv2.resize(face, (160, 160))
            # Apply consistent preprocessing (prewhiten, same as training)
            face = prewhiten(face.astype(np.float32))
            face = np.expand_dims(face, axis=0)

            # Compute embedding
            emb = facenet_model.calculate_embeddings(face)

            # Predict
            probs = model.predict_proba(emb)[0]
            idx = np.argmax(probs)
            conf = probs[idx]

            if conf >= CONFIDENCE_THRESHOLD:
                name = label_encoder.inverse_transform([idx])[0]
            else:
                name = "  "

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"Detected {name} ({conf:.2f})")

    cv2.imshow("Video Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
sess.close()
print("Video processing complete.")