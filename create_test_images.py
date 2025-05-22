import cv2
import numpy as np
import os

def create_test_images():
    # Create test_faces.jpg
    faces_img = np.zeros((500, 800, 3), dtype=np.uint8)
    faces_img.fill(255)  # White background
    
    # Draw some faces
    cv2.circle(faces_img, (200, 200), 50, (0, 255, 0), -1)  # Green face (Employee)
    cv2.circle(faces_img, (400, 200), 50, (0, 0, 255), -1)  # Red face (Criminal)
    cv2.circle(faces_img, (600, 200), 50, (128, 128, 128), -1)  # Gray face (Unknown)
    
    # Add some features
    for x in [200, 400, 600]:
        # Eyes
        cv2.circle(faces_img, (x-15, 185), 5, (0, 0, 0), -1)
        cv2.circle(faces_img, (x+15, 185), 5, (0, 0, 0), -1)
        # Mouth
        cv2.ellipse(faces_img, (x, 220), (20, 10), 0, 0, 180, (0, 0, 0), 2)
    
    # Create test_weapons.jpg
    weapons_img = np.zeros((500, 800, 3), dtype=np.uint8)
    weapons_img.fill(255)  # White background
    
    # Draw some weapons
    # Gun
    cv2.rectangle(weapons_img, (200, 200), (300, 250), (0, 0, 0), -1)
    cv2.rectangle(weapons_img, (300, 220), (350, 230), (0, 0, 0), -1)
    
    # Knife
    cv2.rectangle(weapons_img, (400, 200), (420, 300), (0, 0, 0), -1)
    cv2.line(weapons_img, (400, 200), (450, 150), (0, 0, 0), 2)
    cv2.line(weapons_img, (420, 200), (450, 150), (0, 0, 0), 2)
    
    # Save images
    cv2.imwrite('test_faces.jpg', faces_img)
    cv2.imwrite('test_weapons.jpg', weapons_img)
    print("Test images created: test_faces.jpg and test_weapons.jpg")

if __name__ == '__main__':
    create_test_images() 