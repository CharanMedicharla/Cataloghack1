import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from overlay import overlay_image

# Initialize MediaPipe face and pose detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load accessories
glasses = Image.open('accessories/glasses.png').convert("RGBA")
hat = Image.open('accessories/hat.png').convert("RGBA")
shirt = Image.open('accessories/shirt.png').convert("RGBA")

# Function to resize and place accessories based on landmarks
def apply_accessories(frame, face_box):
    x_min = int(face_box.xmin * frame.shape[1])
    y_min = int(face_box.ymin * frame.shape[0])
    width = int(face_box.width * frame.shape[1])
    height = int(face_box.height * frame.shape[0])

    # Convert OpenCV frame to PIL image
    frame_pil = Image.fromarray(frame)

    # Resize accessories to fit the face size
    glasses_resized = glasses.resize((width, int(height * 0.4)))
    hat_resized = hat.resize((width, int(height * 0.6)))
    
    # Overlay accessories
    frame_pil = overlay_image(frame_pil, glasses_resized, x_min, y_min + int(height * 0.3))
    frame_pil = overlay_image(frame_pil, hat_resized, x_min, y_min - int(height * 0.6))

    # Convert PIL image back to OpenCV format
    frame = np.array(frame_pil)
    return frame

# Start face detection loop
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Convert the BGR frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = face_detection.process(frame_rgb)
        
        # Draw face landmarks and apply accessories
        if results.detections:
            for detection in results.detections:
                face_box = detection.location_data.relative_bounding_box
                frame = apply_accessories(frame, face_box)
        
        # Display the output frame
        cv2.imshow('Virtual Try-On', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing ESC
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
