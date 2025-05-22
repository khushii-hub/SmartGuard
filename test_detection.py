import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from detection_client import DetectionClient
import os
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the correct model directory path
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'face_recognition_package', 'models')

# Load the trained model
try:
    detector = DetectionClient(model_dir=model_dir)
    logger.info("Face detection model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load face detection model: {str(e)}")
    logger.error(traceback.format_exc())
    detector = None

def detect_faces(image_data, min_confidence=0.6):
    try:
        if detector is None:
            logger.error("Face detection model not loaded")
            return {
                "detected": False,
                "message": "Face detection model not loaded"
            }

        # Decode base64 image
        try:
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            logger.info(f"Processing image of size: {frame.shape}")
        except Exception as e:
            logger.error(f"Failed to decode image: {str(e)}")
            return {
                "detected": False,
                "message": "Invalid image data"
            }

        # Run detection
        try:
            faces, _ = detector.process_frame(frame)
            logger.info(f"Raw face detection found {len(faces)} faces")
        except Exception as e:
            logger.error(f"Face detection failed: {str(e)}")
            return {
                "detected": False,
                "message": "Face detection failed"
            }

        if not faces:
            return {
                "detected": False,
                "message": "No faces detected in the image"
            }

        # Filter faces by confidence threshold
        filtered_faces = []
        for face in faces:
            confidence = face.get('confidence', 0)
            if confidence >= min_confidence:
                filtered_faces.append(face)
            else:
                logger.info(f"Face filtered out due to low confidence: {confidence:.2f} < {min_confidence}")

        if not filtered_faces:
            return {
                "detected": False,
                "message": f"No faces detected with confidence >= {min_confidence}"
            }

        # Return info of all faces
        results = []
        for face in filtered_faces:
            x, y, w, h = face['box']
            name = face['name']
            role = face['role']
            alert = face['alert']
            confidence = face.get('confidence', 0)
            label = f"{role}: {name} (Confidence: {confidence:.2f})"
            
            logger.info(f"Detected face: {label} at position ({x}, {y}, {w}, {h})")

            results.append({
                "box": [x, y, w, h],
                "name": name,
                "role": role,
                "alert": alert,
                "label": label,
                "confidence": confidence
            })

        return {
            "detected": True,
            "object_type": "Face",
            "faces": results,
            "message": f"{len(filtered_faces)} face(s) detected with confidence >= {min_confidence}"
        }

    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "detected": False,
            "message": f"Error: {str(e)}"
        }
