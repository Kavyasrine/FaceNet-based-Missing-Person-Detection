import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import facenet

def load_classifier(path="./class/classifier.pkl"):
    """Load trained SVM classifier."""
    if not os.path.exists(path):
        print("Error: classifier.pkl not found.")
        return None, None

    with open(path, "rb") as f:
        data = pickle.load(f)
        clf = data["model"]
        le = data["le"]

    print("Classifier loaded successfully.")
    return clf, le


def load_facenet_model(model_path="./model/20170511-185253.pb"):
    """Load FaceNet graph into memory."""
    if not os.path.exists(model_path):
        print("Error: FaceNet .pb file not found.")
        return None, None, None

    print("Loading FaceNet model...")
    facenet.load_model(model_path)

    graph = tf.compat.v1.get_default_graph()
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

    return images_placeholder, embeddings, phase_train_placeholder


def get_embedding(img_path, sess, images_ph, embeddings, phase_ph):
    """Convert face image into 128-D FaceNet embedding with consistent preprocessing."""
    try:
        # Load and preprocess image consistently with training (resize, convert to RGB, prewhiten)
        img = Image.open(img_path).convert("RGB").resize((160, 160))
        img = np.array(img)
        img = facenet.prewhiten(img)  # Consistent normalization (mean=0, std=1)
        img = np.expand_dims(img, axis=0)

        feed = {images_ph: img, phase_ph: False}
        emb = sess.run(embeddings, feed_dict=feed)
        return emb[0]

    except Exception as e:
        print(f"Error embedding {img_path}: {e}")
        return None


def classify_face(img_path, confidence_threshold=0.5):
    """Return predicted person name and confidence, with threshold for unknowns."""
    clf, le = load_classifier()
    if clf is None:
        return "Classifier not found", 0.0

    images_ph, embeddings, phase_ph = load_facenet_model()
    if images_ph is None:
        return "Model not found", 0.0

    with tf.compat.v1.Session() as sess:
        emb = get_embedding(img_path, sess, images_ph, embeddings, phase_ph)
        if emb is None:
            return "Face not detected or error in processing", 0.0

        probs = clf.predict_proba([emb])[0]
        pred = np.argmax(probs)
        confidence = probs[pred]

        if confidence >= confidence_threshold:
            name = le.inverse_transform([pred])[0]
        else:
            name = "Unknown"

        return name, round(confidence * 100, 2)


if __name__ == "__main__":
    path = input("Enter image path: ")
    name, conf = classify_face(path)
    print("Prediction:", name, "| Confidence:", conf, "%")