from face_recognition_module import FaceRecognitionSystem
import cv2
import time

def main():
    # Initialize the face recognition system
    face_system = FaceRecognitionSystem()
    
    # Example 1: Process a single image
    def process_image(image_path):
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Process frame
        faces, _ = face_system.process_frame(frame)
        
        # Draw results
        for face in faces:
            x, y, w, h = face['box']
            role = face['role']
            name = face['name']
            
            # Draw bounding box
            color = (0, 0, 255) if role == 'Criminal' else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{role}: {name}"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save result
        output_path = "result.jpg"
        cv2.imwrite(output_path, frame)
        print(f"Result saved to {output_path}")
    
    # Example 2: Process video stream
    def process_video(video_path=0):  # 0 for webcam
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                faces, _ = face_system.process_frame(frame)
                
                # Draw results
                for face in faces:
                    x, y, w, h = face['box']
                    role = face['role']
                    name = face['name']
                    
                    # Draw bounding box
                    color = (0, 0, 255) if role == 'Criminal' else (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label
                    label = f"{role}: {name}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Show frame
                cv2.imshow('Face Recognition', frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    # Example usage
    print("1. Process single image")
    print("2. Process video stream")
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        image_path = input("Enter image path: ")
        process_image(image_path)
    elif choice == "2":
        video_path = input("Enter video path (or press Enter for webcam): ")
        if not video_path:
            video_path = 0
        process_video(video_path)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 