import os
import shutil
import cv2
import numpy as np

def create_directory_structure():
    # Create main directories
    os.makedirs('faces/Criminals', exist_ok=True)
    os.makedirs('faces/Employees', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("Created directory structure")

def preprocess_image(img):
    """Preprocess image to improve face detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    return gray

def process_image(image_path, role, person_name):
    """Process a single image and save it in the correct format"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return False
    
    # Preprocess image
    gray = preprocess_image(img)
    
    # Detect face with more lenient parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # More lenient scale factor
        minNeighbors=3,   # Fewer neighbors required
        minSize=(30, 30), # Smaller minimum size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        print(f"No face detected in: {image_path}")
        return False
    
    # Create person directory
    person_dir = os.path.join('faces', role, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face_img = gray[y:y+h, x:x+w]
        # Resize to standard size
        face_img = cv2.resize(face_img, (100, 100))
        # Save image
        output_path = os.path.join(person_dir, f'face_{i}.jpg')
        cv2.imwrite(output_path, face_img)
        print(f"Saved face to: {output_path}")
    
    return True

def normalize_role(role):
    """Convert role to standard format (Criminals or Employees)"""
    role = role.lower()
    if role == 'criminals' or role == 'criminal':
        return 'Criminals'
    elif role == 'employees' or role == 'employee':
        return 'Employees'
    return None

def main():
    # Create directory structure
    create_directory_structure()
    
    # Use the provided dataset path
    dataset_path = r"C:\PS 2\smart_surveillance_flask\face-dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist")
        return
    
    print(f"Processing dataset from: {dataset_path}")
    
    # Process each image in the dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Get role and person name from directory structure
                rel_path = os.path.relpath(root, dataset_path)
                parts = rel_path.split(os.sep)
                
                if len(parts) >= 2:
                    role = normalize_role(parts[0])
                    person_name = parts[1]
                    
                    if role:
                        image_path = os.path.join(root, file)
                        print(f"\nProcessing: {image_path}")
                        process_image(image_path, role, person_name)
                    else:
                        print(f"Invalid role in path: {rel_path} - skipping")
                else:
                    print(f"Invalid directory structure: {rel_path} - skipping")

if __name__ == '__main__':
    main() 