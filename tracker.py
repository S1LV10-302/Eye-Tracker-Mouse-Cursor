import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False  # Disable fail-safe (use carefully)

# Initialize MediaPipe Face Mesh and Hands
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Set up webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Initial values
sensitivity = 5
smooth_factor = 0.5

# Previous position
prev_x, prev_y = pyautogui.position()

# Mouth open detection
MOUTH_OPEN_THRESHOLD = 15  # Pixels
mouth_opened = False

# Tracking control
tracking_enabled = True  # Start with tracking enabled
hand_gesture_detected = False  # To avoid toggling multiple times quickly

# Create trackbars in the main window
cv2.namedWindow("Face Mouse Controller")
cv2.createTrackbar("Sensitivity", "Face Mouse Controller", 1, 20, lambda x: None)
cv2.createTrackbar("Smoothness", "Face Mouse Controller", 4, 20, lambda x: None)

# Set initial positions
cv2.setTrackbarPos("Sensitivity", "Face Mouse Controller", sensitivity)
cv2.setTrackbarPos("Smoothness", "Face Mouse Controller", int(smooth_factor * 10))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get current trackbar values
    sensitivity = cv2.getTrackbarPos("Sensitivity", "Face Mouse Controller")
    smooth_factor = cv2.getTrackbarPos("Smoothness", "Face Mouse Controller") / 10.0

    # Process Face Mesh and Hands
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    # --- Check for hand gesture (to toggle tracking) ---
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Count extended fingers
            fingers_up = 0

            # Thumb (landmarks 4 and 2)
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
                fingers_up += 1

            # Other fingers (landmarks: tip vs pip)
            for tip_idx, pip_idx in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                if hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y:
                    fingers_up += 1

            # If 5 fingers are up → toggle
            if fingers_up == 5:
                if not hand_gesture_detected:
                    tracking_enabled = not tracking_enabled
                    hand_gesture_detected = True
            else:
                hand_gesture_detected = False
    else:
        hand_gesture_detected = False

    # --- Only do face tracking if tracking is enabled ---
    if tracking_enabled and face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            # Nose tip
            nose_tip = landmarks.landmark[1]
            nose_x = int(nose_tip.x * frame_width)
            nose_y = int(nose_tip.y * frame_height)

            cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1)

            norm_x = nose_tip.x
            norm_y = nose_tip.y

            screen_x = (norm_x - 0.5) * 2 * sensitivity
            screen_y = (norm_y - 0.5) * 2 * sensitivity

            screen_x = max(min(screen_x, 1), -1)
            screen_y = max(min(screen_y, 1), -1)

            target_x = (screen_x + 1) / 2 * screen_width
            target_y = (screen_y + 1) / 2 * screen_height

            final_x = prev_x + (target_x - prev_x) * smooth_factor
            final_y = prev_y + (target_y - prev_y) * smooth_factor

            pyautogui.moveTo(final_x, final_y, duration=0.01)

            prev_x, prev_y = final_x, final_y

            # Mouth landmarks
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]

            upper_lip_y = int(upper_lip.y * frame_height)
            lower_lip_y = int(lower_lip.y * frame_height)

            mouth_distance = abs(lower_lip_y - upper_lip_y)

            upper_lip_x = int(upper_lip.x * frame_width)
            lower_lip_x = int(lower_lip.x * frame_width)
            cv2.line(frame, (upper_lip_x, upper_lip_y), (lower_lip_x, lower_lip_y), (255, 0, 0), 2)

            if mouth_distance > MOUTH_OPEN_THRESHOLD:
                if not mouth_opened:
                    pyautogui.click()
                    mouth_opened = True
            else:
                mouth_opened = False

    # Display status information
    status_text = f"Tracking: {'ON' if tracking_enabled else 'PAUSED'}"
    cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0) if tracking_enabled else (0, 0, 255), 2)
    
    settings_text = f"Sensitivity: {sensitivity} | Smoothness: {smooth_factor:.1f}"
    cv2.putText(frame, settings_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 255), 1)
    
    cv2.putText(frame, "Show 5 fingers to toggle tracking", (20, frame_height - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "Press 'q' to quit", (20, frame_height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, "Open mouth to make a click", (20, frame_height - 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Face Mouse Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()