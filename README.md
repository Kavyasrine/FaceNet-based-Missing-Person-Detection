# FaceNet-based-Missing-Person-Detection
A face detection and recognition system using MTCNN for face detection, FaceNet for embedding extraction, and SVM for classification. Fully compatible with Python 3.11 and TensorFlow 2.10.

Overview

This project identifies individuals from images and videos by combining deep learning based embeddings with a traditional classifier. MTCNN detects and aligns faces, FaceNet converts aligned faces into numerical embeddings, and an SVM classifier predicts the identity. The system supports image input, video files, and webcam streams. It is designed for academic, research, and prototype-level face recognition applications.

Features

Face detection using MTCNN
Face alignment
Embedding generation with FaceNet
Identity prediction using SVM
Image and video recognition
High accuracy on controlled datasets
Compatible with Python 3.11

System Requirements

Windows 10 or 11
Python 3.11
TensorFlow 2.10
CPU processing

Installation

Create virtual environment

python -m venv venv
venv\Scripts\activate


Install dependencies

pip install numpy
pip install scipy
pip install opencv-python
pip install mtcnn
pip install scikit-learn
pip install pillow
pip install tensorflow==2.10
pip install protobuf==3.20
pip install joblib

Project Structure
project/
 ├── train_img/
 ├── model/
 │      20170511-185253.pb
 ├── classifier/
 │      classifier.pkl
 ├── detect_face.py
 ├── preprocess.py
 ├── facenet.py
 ├── classifier.py
 ├── identify_face_image.py
 ├── identify_face_video.py
 ├── train_main.py
 └── venv/

Dataset Format

Inside train_img place one folder per person.
The folder name becomes the class label.

train_img/
 ├── person1/
 ├── person2/
 └── person3/


Each folder should contain multiple clear face images.

Training the Classifier

Run:

python train_main.py


This performs dataset loading, face detection, alignment, embedding extraction, SVM training, and saves classifier.pkl.

Running Image Recognition
python identify_face_image.py


Displays identity and confidence.

Running Video Recognition
python identify_face_video.py


Performs realtime detection and recognition on video or webcam.

Working Principle

MTCNN detects the face and extracts bounding boxes.
Preprocessing aligns faces to a fixed size.
FaceNet embeds each face into a 128 dimensional vector.
SVM classifier compares embeddings with trained data.
The system outputs the predicted identity with confidence score.

Technologies Used

MTCNN for face detection
FaceNet for embedding extraction
Scikit learn SVM for classification
OpenCV for video and image handling
TensorFlow for running the FaceNet model
NumPy for numerical operations

Notes

Use clear face images for best accuracy.
Stable lighting improves detection.
Keep classifier.pkl and FaceNet model path correct.
