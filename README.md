<<<<<<< HEAD
# smart_surveillance_flask
=======
# Smart Surveillance Flask System

## Features
- **Face Recognition** (OpenCV LBPH, local dataset)
- **Weapon Detection** (MobileNet SSD + Feature-based matching)
- **REST API** for browser integration
- **SQLite Database** for alert logging

## API Documentation

### Endpoints

#### POST /api/detect
Detect faces and weapons in an image.

**Request:**
```json
{
    "image": "base64_encoded_image"
}
```

**Response:**
```json
{
    "faces": [
        {
            "box": {
                "x": 100,
                "y": 200,
                "width": 150,
                "height": 150
            },
            "label": "Employee",
            "name": "John Doe",
            "confidence": 0.95,
            "alert": false
        }
    ],
    "weapons": [
        {
            "box": {
                "x": 300,
                "y": 400,
                "width": 100,
                "height": 50
            },
            "label": "gun",
            "confidence": 0.85,
            "method": "ssd"
        }
    ]
}
```

#### GET /api/alerts
Get recent alerts.

**Response:**
```json
[
    {
        "type": "face",
        "label": "Criminal",
        "role": "Criminal",
        "name": "Unknown",
        "confidence": 0.95,
        "timestamp": "2024-03-20 12:34:56"
    }
]
```

## Directory Structure
```
smart_surveillance_flask/
├── app.py
├── train_face_recognizer.py
├── detection_client.py
├── requirements.txt
├── README.md
├── /models/
│   ├── face_recognizer.yml
│   ├── label_map.json
│   ├── MobileNetSSD_deploy.caffemodel
│   ├── MobileNetSSD_deploy.prototxt
│   └── /weapons/
│       ├── gun/
│       └── knife/
├── /faces/
│   ├── Criminals/
│   └── Employees/
├── /templates/
│   └── index.html
├── /static/
│   ├── style.css
│   └── main.js
```

## 1. Training the Face Model
1. Place face images in `faces/Criminals/personX/` and `faces/Employees/personY/` (use grayscale, clear faces).
2. Run:
   ```bash
   python train_face_recognizer.py
   ```
   This creates `models/face_recognizer.yml` and `models/label_map.json`.

## 2. Setting Up Weapon Detection
1. For MobileNet SSD:
   - Download and place in `models/`:
     - [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel)
     - [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt)

2. For Feature-based detection:
   - Create `models/weapons/` directory
   - Add weapon images in subdirectories (e.g., `models/weapons/gun/`, `models/weapons/knife/`)
   - System will automatically extract and store features

## 3. Running the API Server
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the server:
   ```bash
   python app.py
   ```
3. The API will be available at `http://localhost:5000`

## 4. Browser Integration
To integrate with your frontend:

1. Face Recognition:
   - Send base64-encoded frames to `/api/detect`
   - Results match face-api.js format for easy integration

2. Weapon Detection:
   - Uses both MobileNet SSD and feature-based matching
   - Results include confidence scores and detection method

Example JavaScript integration:
```javascript
async function processFrame(videoElement) {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    const base64Image = canvas.toDataURL('image/jpeg').split(',')[1];

    const response = await fetch('http://localhost:5000/api/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: base64Image })
    });

    const detections = await response.json();
    // Process detections.faces and detections.weapons
}
```

## 5. Notes
- Works on CPU-only systems
- Alerts are saved in `alerts.db` (SQLite)
- Modular: `DetectionClient` can be reused for other applications

## 6. License
MIT 

# Face Recognition Module

This module provides a simple interface for face recognition using OpenCV's LBPH face recognizer. It can be used to detect and recognize faces in images or video streams.

## Files

1. `face_recognition_module.py` - The main module containing the face recognition system
2. `example_usage.py` - Example script showing how to use the module
3. `models/face_recognizer.yml` - Trained face recognition model
4. `models/label_map.json` - Mapping of labels to names and roles

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- OpenCV Contrib (`pip install opencv-contrib-python`)

## Usage

### Basic Usage

```python
from face_recognition_module import FaceRecognitionSystem

# Initialize the system
face_system = FaceRecognitionSystem()

# Process a frame
faces, _ = face_system.process_frame(frame)

# Each face contains:
# - box: (x, y, w, h) coordinates
# - role: 'Criminal', 'Employee', or 'Unknown'
# - name: Name of the person or 'Unknown'
# - confidence: Recognition confidence score
# - alert: True if role is 'Criminal' or 'Unknown'
```

### Example Script

Run the example script to test the system:

```bash
python example_usage.py
```

This will give you options to:
1. Process a single image
2. Process a video stream (including webcam)

## Model Files

The system requires two model files:
1. `models/face_recognizer.yml` - The trained face recognition model
2. `models/label_map.json` - The mapping of labels to names and roles

Make sure these files are present in the `models` directory.

## Known People

The system currently recognizes:
- Harsha (Employee)
- Khushii (Criminal)

All other faces will be marked as "Unknown".

## Integration

To integrate this module into another project:

1. Copy `face_recognition_module.py` to your project
2. Copy the `models` directory to your project
3. Import and use the `FaceRecognitionSystem` class as shown in the example

## Notes

- The system uses OpenCV's LBPH face recognizer
- Face detection is done using Haar cascades
- The system includes preprocessing steps for better recognition
- Recognition results include bounding boxes, roles, and names 
>>>>>>> fda3aab (Initial commit)
