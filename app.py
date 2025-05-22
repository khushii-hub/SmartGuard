import os
import cv2
import numpy as np
import pickle
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from test_detection import detect_faces  # Assuming you have the face detection logic here
from datetime import datetime
import time
import sys
import base64
from io import BytesIO

# Configure logging with more detail but cleaner output
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS with more specific configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000", "*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "expose_headers": ["Content-Type", "Authorization"]
    }
})

# Load KNN model for face recognition
with open("trained_knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

# Load face detector model
MODEL_PATH = "models"
PROTOTXT_PATH = os.path.join(MODEL_PATH, "deploy.prototxt")
WEIGHTS_PATH = os.path.join(MODEL_PATH, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)

# Function to decode base64 image to OpenCV format
def decode_base64_image(image_data):
    """Decode base64 string to OpenCV image format"""
    try:
        # Remove the prefix 'data:image/jpeg;base64,' if it exists
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        img_data = base64.b64decode(image_data)
        img = np.array(bytearray(img_data), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        return None

# Face detection function
def detect_faces(image):
    """Detect faces and return coordinates and faces"""
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            face = image[y1:y2, x1:x2]
            if face.size != 0:
                face = cv2.resize(face, (100, 100))  # Resize to fixed size for KNN
                faces.append(face)
                boxes.append((x1, y1, x2, y2))
    return faces, boxes

# Feature extraction function for face recognition
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.flatten()

@app.route('/test', methods=['GET'])
def test():
    """Simple test endpoint to verify server is responding"""
    logger.info("Test endpoint called")
    return jsonify({
        'status': 'ok',
        'message': 'Backend is responding',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/detect', methods=['POST', 'OPTIONS'])
def detect():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Add timeout handling
        start_time = time.time()

        # Log the raw request data for debugging
        logger.info(f"Raw request data: {request.get_data()}")

        # Validate request
        if not request.is_json:
            logger.warning("Request is not JSON")
            return jsonify({
                'detected': False,
                'message': 'Request must be JSON',
                'timestamp': datetime.now().isoformat()
            }), 400
            
        data = request.json
        if not data:
            logger.warning("No JSON data in request body")
            return jsonify({
                'detected': False,
                'message': 'No JSON data provided',
                'timestamp': datetime.now().isoformat()
            }), 400
            
        if 'image' not in data:
            logger.warning("No image data in request JSON")
            return jsonify({
                'detected': False,
                'message': 'No image data provided',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Validate image data
        image_data = data['image']
        if not isinstance(image_data, str):
            logger.warning("Image data is not a string")
            return jsonify({
                'detected': False,
                'message': 'Invalid image data format',
                'timestamp': datetime.now().isoformat()
            }), 400
            
        if not image_data.startswith('data:image'):
            logger.warning("Image data does not start with data:image")
            return jsonify({
                'detected': False,
                'message': 'Invalid image data format',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get minimum confidence threshold
        min_confidence = float(data.get('minConfidence', 0.6))

        # Decode the base64 image
        img = decode_base64_image(image_data)
        if img is None:
            return jsonify({
                'detected': False,
                'message': 'Invalid image format',
                'timestamp': datetime.now().isoformat()
            }), 400

        # Process the image with our face detection function
        logger.info("Processing image data...")
        
        try:
            faces, boxes = detect_faces(img)
            logger.info(f"Detection complete: {len(faces)} faces found")
            
            if faces:
                # Assume only one face for simplicity, you can adjust as needed
                face_image = faces[0]
                name = "Unknown"  # Default name if no match found
                confidence = 0

                # Predict the face using KNN
                face_feature = extract_features(face_image)
                predicted_name = knn_model.predict([face_feature])
                name = predicted_name[0]
                confidence = knn_model.score([face_feature])  # Confidence score

                return jsonify({
                    'detected': True,
                    'name': name,
                    'confidence': confidence,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'Face detected'
                })

            else:
                return jsonify({
                    'detected': False,
                    'timestamp': datetime.now().isoformat(),
                    'message': 'No face detected'
                })
                
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'detected': False,
                'timestamp': datetime.now().isoformat(),
                'message': f'Face detection error: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'detected': False,
            'timestamp': datetime.now().isoformat(),
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        model_dir = os.path.join(os.path.dirname(__file__), 'face_recognition_package', 'models')
        label_map_path = os.path.join(model_dir, 'label_map.json')
        
        if not os.path.exists(label_map_path):
            return jsonify({
                'status': 'error',
                'message': 'Model files not found',
                'timestamp': datetime.now().isoformat()
            }), 500
            
        return jsonify({
            'status': 'ok',
            'message': 'Server is running',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Health check failed: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server for face detection...")
        logger.info("API endpoint: http://127.0.0.1:5002/detect")
        logger.info("Test endpoint: http://127.0.0.1:5002/test")
        logger.info("Health check: http://127.0.0.1:5002/health")
        
        app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
