import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# --------------------------- IMAGE UTILITIES --------------------------- #

def to_rgb(img):
    """Convert grayscale to RGB."""
    if len(img.shape) == 2:
        w, h = img.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
        return ret
    return img

def prewhiten(x):
    """Normalize image for FaceNet (mean=0, std=1)."""
    mean = np.mean(x)
    std = np.maximum(np.std(x), 1.0 / np.sqrt(x.size))
    return (x - mean) / std

def flip(image, random_flip=True):
    """Random horizontal flip."""
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

# --------------------------- MODEL WRAPPER --------------------------- #

class Facenet:
    def __init__(self, model_path='./model/20170511-185253.pb'):
        self.model_path = model_path
        self.sess = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None

    def load_model(self):
        """Load FaceNet frozen graph (.pb) into TF1 session."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"FaceNet model not found: {self.model_path}")

        print(f"Loading FaceNet model from: {self.model_path}")
        tf.compat.v1.disable_eager_execution()
        self.sess = tf.compat.v1.Session()
        with gfile.FastGFile(self.model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

        # Cache tensor references for efficiency
        graph = tf.compat.v1.get_default_graph()
        self.images_placeholder = graph.get_tensor_by_name("input:0")
        self.embeddings = graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        print("Model loaded successfully.")
        return self.sess

    def calculate_embeddings(self, images):
        """Compute 128-D embeddings for preprocessed face images (expects batch of shape [N, 160, 160, 3])."""
        if self.sess is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
        emb = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb

    def preprocess_and_embed(self, img_path_or_array):
        """Convenience method: Load/preprocess image and compute embedding (for single images)."""
        if isinstance(img_path_or_array, str):
            # Load from path
            from PIL import Image
            img = Image.open(img_path_or_array).convert("RGB").resize((160, 160))
            img = np.array(img)
        else:
            img = img_path_or_array  # Assume already loaded array

        # Apply consistent preprocessing
        img = prewhiten(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)

        return self.calculate_embeddings(img)[0]  # Return single embedding