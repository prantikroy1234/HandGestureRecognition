import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)

# Gesture classification function
def get_finger_status(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_status = []

    landmarks = hand_landmarks.landmark

    for tip in finger_tips:
        # Tip is above PIP joint → finger is open
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_status.append(1)
        else:
            finger_status.append(0)

    # Thumb (x-axis logic)
    if landmarks[4].x < landmarks[3].x:
        thumb = 1
    else:
        thumb = 0

    finger_status.insert(0, thumb)  # [Thumb, Index, Middle, Ring, Pinky]
    return finger_status

def get_gesture_name(fingers):
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist ✊"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm ✋"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up 👍"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace ✌️"
    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing ☝️"
    elif fingers == [0, 1, 1, 1, 1]:
        return "Four Fingers 🖖"
    elif fingers == [0, 0, 0, 0, 1]:
        return "Pinky 🤙"
    else:
        return "Unknown 🤔"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers = get_finger_status(hand_landmarks)
            gesture = get_gesture_name(fingers)

            # Show gesture on screen
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
