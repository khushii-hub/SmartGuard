import face_recognition
import cv2
import os
import numpy as np

KNOWN_FACES_DIR = 'face_dataset'
TOLERANCE = 0.5

known_faces = []
known_names = []

for label in os.listdir(KNOWN_FACES_DIR):
    for img_path in os.listdir(os.path.join(KNOWN_FACES_DIR, label)):
        image = face_recognition.load_image_file(os.path.join(KNOWN_FACES_DIR, label, img_path))
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(label)

def process_frame_for_faces(frame):
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, encoding, TOLERANCE)
        name = "Unknown"
        if True in matches:
            index = matches.index(True)
            name = known_names[index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame
