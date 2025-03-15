import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load trained Random Forest model with error handling
MODEL_PATH = "sign_model_rf.pkl"
WORDS = ["YES", "NO", "THANKYOU", "SORRY", "HELLO", "I LOVE YOU", "GOODBYE", "PLEASE", "YOU ARE WELCOME", "FAMILY",
         "HOUSE", "LOVE"]  # Updated word list
SEQUENCE_LENGTH = 50  # Adjusted for 3-second sequences
FEATURE_DIM = 42  # 21 landmarks * 2 (x, y)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit()

model = joblib.load(MODEL_PATH)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)
sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    landmarks = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        landmarks = [0] * FEATURE_DIM  # If no hand detected, fill with zeros

    sequence.append(landmarks)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    if len(sequence) == SEQUENCE_LENGTH:
        feature_vector = np.array(sequence).flatten()  # Flatten sequence into a 1D feature vector
        prediction = model.predict([feature_vector])
        predicted_word = WORDS[prediction[0]]

        cv2.putText(frame, f"{predicted_word}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()