import os
import cv2
import numpy as np
import json

DATASET_DIR = 'face-dataset'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'face_recognizer.yml')

LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'label_map.json')
IMAGE_SIZE = (200, 200)  # uniform resize

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_images_and_labels(dataset_dir):
    image_paths = []
    labels = []
    label_map = {}
    current_label = 0

    role_mapping = {
        'criminals': 'Criminals',
        'employees': 'Employees'
    }

    for role_dir in os.listdir(dataset_dir):
        role_path = os.path.join(dataset_dir, role_dir)
        if not os.path.isdir(role_path):
            continue

        proper_role = role_mapping.get(role_dir.lower(), role_dir)

        for person in os.listdir(role_path):
            person_path = os.path.join(role_path, person)
            if not os.path.isdir(person_path):
                continue

            label_map[current_label] = {'role': proper_role, 'name': person}

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                image_paths.append(img_path)
                labels.append(current_label)

            current_label += 1

    return image_paths, labels, label_map

def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.equalizeHist(img)  # normalize contrast
    img = cv2.resize(img, IMAGE_SIZE)  # consistent size
    return img

def main():
    print('Loading images...')
    image_paths, labels, label_map = get_images_and_labels(DATASET_DIR)
    faces = []
    y_labels = []

    for img_path, label in zip(image_paths, labels):
        img = preprocess_image(img_path)
        if img is None:
            print(f'Warning: Could not read {img_path}')
            continue
        faces.append(img)
        y_labels.append(label)

    print(f'Training on {len(faces)} images...')
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(y_labels))
    recognizer.save(MODEL_PATH)

    with open(LABEL_MAP_PATH, 'w') as f:
        json.dump(label_map, f, indent=4)

    print(f'Model saved to {MODEL_PATH}')
    print(f'Label map saved to {LABEL_MAP_PATH}')

if __name__ == '__main__':
    main()
