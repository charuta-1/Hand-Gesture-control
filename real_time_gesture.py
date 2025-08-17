import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("gesture_model.keras")  # Make sure this path is correct

# Label map for prediction
label_map = {
    0: "Palm",
    1: "L",
    2: "Fist",
    3: "Fist_moved",
    4: "Thumb",
    5: "Index",
    6: "Ok",
    7: "Palm_moved",
    8: "C",
    9: "Down"
}

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)

    # Predict
    prediction = model.predict(reshaped)
    predicted_class = np.argmax(prediction)
    gesture = label_map[predicted_class]

    # Display the result
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Gesture Recognition", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
