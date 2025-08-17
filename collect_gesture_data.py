import cv2
import csv
import os
import mediapipe as mp

# === Setup ===
GESTURE_LABEL = "STOP"  # Change this for each gesture you collect
SAVE_PATH = "gesture_data.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# === Collect & Save Landmarks ===
with open(SAVE_PATH, mode='a', newline='') as f:
    csv_writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks: x, y, z
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z]) # Append each landmark's x, y, z
                row.append(GESTURE_LABEL)  # Add gesture label at the end
                csv_writer.writerow(row)   # Save to CSV

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Recording: {GESTURE_LABEL}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

