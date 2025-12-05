# train_main.py – Windows + Python 3.11 + TensorFlow 2.x compatible
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Alternative classifier for small data
from PIL import Image
import tensorflow as tf
import facenet  # your facenet.py
from facenet import prewhiten, flip  # For augmentation

# -------------------- CONFIG --------------------
PRE_IMG_DIR = r"C:\Users\rkavy\Downloads\PersonMissed\Full project_face recog\pre_img"
MODEL_PATH = r"C:\Users\rkavy\Downloads\PersonMissed\model\20170511-185253.pb"
CLASSIFIER_PATH = r"C:\Users\rkavy\Downloads\PersonMissed\class\classifier.pkl"
USE_KNN = False  # Set to True to use KNN instead of SVM (better for small datasets)
AUGMENT_DATA = True  # Enable data augmentation to expand small dataset

# Disable eager execution for TF1 .pb model
tf.compat.v1.disable_eager_execution()

# -------------------- LOAD MODEL --------------------
print("Loading FaceNet model...")
facenet_model = facenet.Facenet(model_path=MODEL_PATH)
sess = facenet_model.load_model()

# -------------------- LOAD DATA --------------------
print("Loading preprocessed faces...")
classes = [d for d in os.listdir(PRE_IMG_DIR) if os.path.isdir(os.path.join(PRE_IMG_DIR, d))]
if len(classes) == 0:
    print("Error: No preprocessed faces found in pre_img.")
    exit()

X, y = [], []

for person in classes:
    person_path = os.path.join(PRE_IMG_DIR, person)
    for file in os.listdir(person_path):
        if not file.lower().endswith(('.npy')):  # Updated to load .npy files from data_preprocess.py
            continue

        img_path = os.path.join(person_path, file)
        try:
            # Load prewhitened array directly (no need to resize/prewhiten again)
            img = np.load(img_path)
            img = np.expand_dims(img, axis=0).astype(np.float32)

            # Compute embedding
            feed_dict = {
                facenet_model.images_placeholder: img,
                facenet_model.phase_train_placeholder: False
            }
            emb = sess.run(facenet_model.embeddings, feed_dict=feed_dict)
            X.append(emb[0])
            y.append(person)
            print(f"Processed: {person}/{file}")

            # Data augmentation (if enabled)
            if AUGMENT_DATA:
                # Horizontal flip
                flipped_img = flip(img[0])  # flip expects [H,W,C]
                flipped_img = np.expand_dims(flipped_img, axis=0)
                feed_dict_flipped = {
                    facenet_model.images_placeholder: flipped_img,
                    facenet_model.phase_train_placeholder: False
                }
                emb_flipped = sess.run(facenet_model.embeddings, feed_dict=feed_dict_flipped)
                X.append(emb_flipped[0])
                y.append(person)
                print(f"Augmented (flip): {person}/{file}")

        except Exception as e:
            print(f"Skipped {file}: {e}")

if len(X) == 0:
    print("Error: No embeddings generated. Check pre_img folder and preprocessing.")
    exit()

X = np.array(X)
print(f"Total embeddings: {X.shape} (after augmentation if enabled)")

# -------------------- TRAIN CLASSIFIER --------------------
print("Encoding labels and training classifier...")
le = LabelEncoder()
y_encoded = le.fit_transform(y)

if USE_KNN:
    clf = KNeighborsClassifier(n_neighbors=min(3, len(classes)))  # KNN for small data
    print("Using KNN classifier.")
else:
    clf = SVC(kernel='linear', probability=True, random_state=42)
    print("Using SVM classifier.")

clf.fit(X, y_encoded)

# -------------------- SAVE CLASSIFIER --------------------
os.makedirs(os.path.dirname(CLASSIFIER_PATH), exist_ok=True)
with open(CLASSIFIER_PATH, 'wb') as f:
    pickle.dump({"model": clf, "le": le}, f, protocol=4)

print(f"Classifier saved to: {CLASSIFIER_PATH}")
print("Training completed successfully!")

# -------------------- CLOSE SESSION --------------------
sess.close()