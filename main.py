import cv2
import mediapipe as mp
import pyttsx3
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


engine = pyttsx3.init()
engine.setProperty('rate', 150) # Set speaking speed speed
last_spoken = "" # Prevents repeating the same word constantly

def calculate_distance(p1, p2):
    
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def recognize_sign(landmarks):
    
    
    
    wrist = landmarks[0]
    
    thumb_open = calculate_distance(landmarks[4], wrist) > calculate_distance(landmarks[2], wrist)
    index_open = calculate_distance(landmarks[8], wrist) > calculate_distance(landmarks[6], wrist)
    middle_open = calculate_distance(landmarks[12], wrist) > calculate_distance(landmarks[10], wrist)
    ring_open = calculate_distance(landmarks[16], wrist) > calculate_distance(landmarks[14], wrist)
    pinky_open = calculate_distance(landmarks[20], wrist) > calculate_distance(landmarks[18], wrist)

    
    if index_open and middle_open and not ring_open and not pinky_open:
        return "Peace"
    elif index_open and not middle_open and not ring_open and pinky_open:
        return "Rock on"
    elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "Thumbs Up"
    elif index_open and middle_open and ring_open and pinky_open and thumb_open:
        return "Hello"
    elif not index_open and not middle_open and not ring_open and not pinky_open:
        return "Fist"
    
    return "Unknown"


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        detection_result = detector.detect_for_video(mp_image, timestamp)
        
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                
                current_sign = recognize_sign(hand_landmarks)
                
                
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                
                
                cv2.putText(frame, f"Sign: {current_sign}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if current_sign != "Unknown" and current_sign != last_spoken:
                    print(f"Speaking: {current_sign}")
                    engine.say(current_sign)
                    engine.runAndWait() # Trigger the speaker audio output
                    last_spoken = current_sign

        cv2.imshow('Sign Language to Voice Converter', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()





