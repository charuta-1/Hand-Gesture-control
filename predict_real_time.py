import cv2
"We use OpenCV to access the webcam and display video."
import numpy as np
"NumPy is used to handle numerical data like arrays."
import mediapipe as mp
"MediaPipe is a library for real-time hand tracking and extracting hand landmarks."
from tensorflow.keras.models import load_model
"This line loads the trained CNN model we saved earlier using Keras."

# Load trained model
model = load_model('gesture_cnn_model.h5')
"We load our trained model, which knows how to classify hand gestures."

# Define gesture labels (update these according to your training data)
labels = ['BACKWARD', 'FORWARD', 'RIGHT', 'LEFT', 'STOP']  
"These are the gesture labels the model can predict. You should match them with the labels used during training."

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
"This sets up MediaPipe to detect hands in video. We allow only 1 hand, and set the confidence thresholds for detection and tracking."



# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    "This loop reads frames from the webcam until the camera is closed or an error occurs."

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    "We flip the frame horizontally so that it feels like a mirror."
    "MediaPipe works with RGB images, so we convert the frame from BGR to RGB."

    # Process frame with MediaPipe
    result = hands.process(rgb)
    "We send the image to MediaPipe, and it returns the hand landmarks if a hand is detected."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            "If a hand is found, we loop through the landmarks to process them."
            # Extract landmarks
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
                "We extract the x, y, and z values for all 21 landmarks and save them in a list."
            landmark_array = np.array(landmark_list).reshape(21, 3)
            "We convert the list into a NumPy array with shape (21, 3) — one row for each landmark."

            # Normalize landmarks
            base_point = landmark_array[0]
            normalized = landmark_array - base_point
            "We normalize the landmarks by subtracting the first landmark (usually the wrist) to make the position relative to the hand."
            normalized = normalized.reshape(21, 3, 1)
            "We reshape the array to match the input shape the CNN model expects."

            # Predict gesture
            prediction = model.predict(np.array([normalized]), verbose=0)
            "We send the normalized landmarks to our trained model to get a prediction."
            class_id = np.argmax(prediction) # Get the index of the class with the highest probability
            class_name = labels[class_id]
            "We get the label name (like 'STOP' or 'LEFT') based on the predicted class index."

            # Draw landmarks and prediction
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            "This draws the landmarks and connections on the hand in the video frame."
            cv2.putText(frame, class_name, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            "We display the predicted gesture name on the screen."


    # Display result
    cv2.imshow("Hand Gesture Recognition", frame)
    "This shows the webcam feed with landmarks and predicted gesture on the screen."
    if cv2.waitKey(1) & 0xFF == 27:  # Press Esc to exit
        break
    "If we press the Esc key, the program stops."

cap.release()
cv2.destroyAllWindows()
"Finally, we release the camera and close all the OpenCV windows."

#python predict_real_time.py