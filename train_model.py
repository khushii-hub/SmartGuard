import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier

# Paths
DATASET_DIR = "face-dataset"
MODEL_PATH = "models"
PROTOTXT_PATH = os.path.join(MODEL_PATH, "deploy.prototxt")
WEIGHTS_PATH = os.path.join(MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel")

# Load OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = image[y1:y2, x1:x2]
            if face.size != 0:
                face = cv2.resize(face, (100, 100))
                faces.append(face)
    return faces

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.flatten()

data = []
labels = []

# Traverse all category folders (criminals, employees)
for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    # Inside each category, traverse person folders
    for person_name in os.listdir(category_path):
        person_path = os.path.join(category_path, person_name)
        if not os.path.isdir(person_path):
            continue

        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue
            faces = detect_face(image)
            for face in faces:
                features = extract_features(face)
                data.append(features)
                labels.append(person_name)

# Train KNN model
print(f"Training on {len(data)} face samples...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, labels)

# Save model
with open("trained_knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

print("✅ Model trained and saved as 'trained_knn_model.pkl'.")
