import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from overlay import overlay_image

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

glasses = Image.open('accessories/glasses.png').convert("RGBA")
hat = Image.open('accessories/hat.png').convert("RGBA")
shirt = Image.open('accessories/shirt.png').convert("RGBA")

def apply_accessories(frame, face_box):
    x_min = int(face_box.xmin * frame.shape[1])
    y_min = int(face_box.ymin * frame.shape[0])
    width = int(face_box.width * frame.shape[1])
    height = int(face_box.height * frame.shape[0])

    frame_pil = Image.fromarray(frame)

    glasses_resized = glasses.resize((width, int(height * 0.4)))
    hat_resized = hat.resize((width, int(height * 0.6)))
    
    frame_pil = overlay_image(frame_pil, glasses_resized, x_min, y_min + int(height * 0.3))
    frame_pil = overlay_image(frame_pil, hat_resized, x_min, y_min - int(height * 0.6))

    frame = np.array(frame_pil)
    return frame

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_detection.process(frame_rgb)
        
        if results.detections:
            for detection in results.detections:
                face_box = detection.location_data.relative_bounding_box
                frame = apply_accessories(frame, face_box)
        
        cv2.imshow('Virtual Try-On', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:  
            break


cap.release()
cv2.destroyAllWindows()
