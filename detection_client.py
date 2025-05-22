import cv2
import os
import numpy as np
import json

class DetectionClient:
    def __init__(self, model_dir='models'):
        # Get the correct model directory path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.path.join(script_dir, 'face_recognition_package', 'models')
        
        print(f"Loading face detection model from: {self.model_dir}")
        
        # Initialize face detector with more lenient parameters
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise Exception("Failed to load face cascade classifier")
        
        # Load training data
        data_path = os.path.join(self.model_dir, 'training_data.npz')
        if os.path.exists(data_path):
            data = np.load(data_path)
            self.faces = data['faces']
            self.labels = data['labels']
            print(f"Loaded {len(self.faces)} training faces")
        else:
            print(f"[WARNING] Training data not found: {data_path}")
            self.faces = []
            self.labels = []
        
        # Load label map
        label_map_path = os.path.join(self.model_dir, 'label_map.json')
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            print(f"Loaded label map with {len(self.label_map)} entries")
        else:
            print(f"[WARNING] Label map not found: {label_map_path}")
            self.label_map = {}

    def preprocess_face(self, face_img):
        """Enhanced face preprocessing for better accuracy"""
        try:
            # Ensure input is uint8
            face_img = face_img.astype(np.uint8)
            
            # Apply adaptive histogram equalization first
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            face_img = clahe.apply(face_img)
            
            # Apply bilateral filter with optimized parameters
            face_img = cv2.bilateralFilter(face_img, 9, 75, 75)
            
            # Apply histogram equalization
            face_img = cv2.equalizeHist(face_img)
            
            # Apply very light Gaussian blur to reduce noise
            face_img = cv2.GaussianBlur(face_img, (3, 3), 0.5)
            
            # Normalize the image
            face_img = cv2.normalize(face_img, None, 0, 255, cv2.NORM_MINMAX)
            
            # Ensure output is uint8
            return face_img.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in face preprocessing: {str(e)}")
            return face_img.astype(np.uint8)  # Return original image if preprocessing fails

    def compare_faces(self, face1, face2):
        """Compare two face images using multiple metrics with voting"""
        try:
            # Ensure both faces are uint8 type
            face1 = face1.astype(np.uint8)
            face2 = face2.astype(np.uint8)
            
            # Preprocess both faces
            face1 = self.preprocess_face(face1)
            face2 = self.preprocess_face(face2)
            
            # Calculate multiple similarity metrics
            metrics = {}
            
            # 1. Pixel difference (normalized)
            pixel_diff = np.mean(np.abs(face1.astype(np.float32) - face2.astype(np.float32)))
            metrics['pixel'] = 1 - (pixel_diff / 255.0)
            
            # 2. Structural similarity index
            try:
                metrics['ssim'] = cv2.matchTemplate(face1, face2, cv2.TM_CCOEFF_NORMED)[0][0]
            except cv2.error:
                metrics['ssim'] = 0.0
            
            # 3. Histogram similarity
            hist1 = cv2.calcHist([face1], [0], None, [256], [0,256])
            hist2 = cv2.calcHist([face2], [0], None, [256], [0,256])
            metrics['hist'] = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # 4. Edge similarity with multiple thresholds
            try:
                edges1_1 = cv2.Canny(face1, 30, 100)
                edges2_1 = cv2.Canny(face2, 30, 100)
                edges1_2 = cv2.Canny(face1, 50, 150)
                edges2_2 = cv2.Canny(face2, 50, 150)
                metrics['edge1'] = cv2.matchTemplate(edges1_1, edges2_1, cv2.TM_CCOEFF_NORMED)[0][0]
                metrics['edge2'] = cv2.matchTemplate(edges1_2, edges2_2, cv2.TM_CCOEFF_NORMED)[0][0]
            except cv2.error:
                metrics['edge1'] = 0.0
                metrics['edge2'] = 0.0
            
            # 5. Local Binary Pattern similarity
            lbp1 = cv2.calcHist([face1], [0], None, [256], [0,256])
            lbp2 = cv2.calcHist([face2], [0], None, [256], [0,256])
            metrics['lbp'] = cv2.compareHist(lbp1, lbp2, cv2.HISTCMP_CORREL)
            
            # 6. Gradient magnitude similarity
            try:
                grad1 = cv2.Laplacian(face1, cv2.CV_64F)
                grad2 = cv2.Laplacian(face2, cv2.CV_64F)
                metrics['grad'] = cv2.matchTemplate(grad1.astype(np.uint8), grad2.astype(np.uint8), cv2.TM_CCOEFF_NORMED)[0][0]
            except cv2.error:
                metrics['grad'] = 0.0
            
            # Voting system
            votes = 0
            total_metrics = len(metrics)
            
            # Thresholds for each metric
            thresholds = {
                'pixel': 0.7,
                'ssim': 0.6,
                'hist': 0.5,
                'edge1': 0.5,
                'edge2': 0.5,
                'lbp': 0.5,
                'grad': 0.5
            }
            
            # Count votes
            for metric, value in metrics.items():
                if value > thresholds[metric]:
                    votes += 1
            
            # Calculate final score
            vote_ratio = votes / total_metrics
            
            # Weighted combination of metrics
            weights = {
                'pixel': 0.25,
                'ssim': 0.25,
                'hist': 0.15,
                'edge1': 0.10,
                'edge2': 0.10,
                'lbp': 0.10,
                'grad': 0.05
            }
            
            weighted_score = sum(metrics[metric] * weights[metric] for metric in metrics)
            
            # Combine voting and weighted score
            final_score = (vote_ratio * 0.4) + (weighted_score * 0.6)
            
            return final_score
            
        except Exception as e:
            print(f"Error in face comparison: {str(e)}")
            return 0.0

    def process_frame(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply basic preprocessing to the entire frame
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # More lenient scale factor
            minNeighbors=3,      # Fewer neighbors required
            minSize=(30, 30),    # Smaller minimum size
            maxSize=(400, 400),  # Larger maximum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Raw face detection found {len(faces)} faces")
        result = []

        for (x, y, w, h) in faces:
            # Add padding around face
            padding = int(min(w, h) * 0.15)  # Moderate padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + w + padding)
            y2 = min(gray.shape[0], y + h + padding)
            
            face_roi = gray[y1:y2, x1:x2]
            face_roi = cv2.resize(face_roi, (100, 100))  # Match training size
            
            # Find best match
            best_match_id = -1
            best_match_score = -1
            second_best_score = -1
            best_match_name = None
            
            # Group scores by person
            person_scores = {}
            
            for i, train_face in enumerate(self.faces):
                score = self.compare_faces(face_roi, train_face)
                label_id = self.labels[i]
                person_info = self.label_map.get(str(label_id), {'name': 'Unknown', 'role': 'Unknown'})
                name = person_info['name']
                
                if name not in person_scores:
                    person_scores[name] = []
                person_scores[name].append(score)
            
            # Calculate scores for each person using multiple methods
            person_final_scores = {}
            for name, scores in person_scores.items():
                # Use multiple methods to determine the final score
                max_score = np.max(scores)
                avg_score = np.mean(scores)
                median_score = np.median(scores)
                
                # Combine scores with weights
                final_score = (max_score * 0.5) + (avg_score * 0.3) + (median_score * 0.2)
                person_final_scores[name] = final_score
                
                if final_score > best_match_score:
                    second_best_score = best_match_score
                    best_match_score = final_score
                    best_match_name = name
                elif final_score > second_best_score:
                    second_best_score = final_score
            
            # Calculate confidence based on difference between best and second best match
            confidence = best_match_score
            if second_best_score > 0:
                confidence = best_match_score - second_best_score
            
            # More lenient threshold for matching with confidence check
            if best_match_score > 0.35 and confidence > 0.1:  # Added confidence threshold
                person_info = next((info for info in self.label_map.values() if info['name'] == best_match_name), 
                                 {'name': 'Unknown', 'role': 'Unknown'})
                name = person_info['name']
                role = person_info['role']
                alert = True if role == 'Criminals' else False
            else:
                name = 'Unknown'
                role = 'Unknown'
                alert = True  # Unknown is always an alert
            
            result.append({
                'box': (x, y, w, h),
                'name': name,
                'role': role,
                'alert': alert,
                'confidence': round(confidence * 100, 2)  # Convert to percentage
            })

        return result, gray
