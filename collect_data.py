import cv2
import os
import numpy as np
import mediapipe as mp

# Define constants
DATA_PATH = "sign_data"  # Folder to save data
WORDS = ["YES", "NO", "THANKYOU", "SORRY", "HELLO", "I LOVE YOU", "GOODBYE", "PLEASE", "YOU ARE WELCOME", "FAMILY",
         "HOUSE", "LOVE"]  # Updated word list
SEQUENCE_LENGTH = 50  # Number of frames per sequence (assuming 30 FPS for 3 sec)
NUM_SEQUENCES = 50  # Number of sequences per word

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Create dataset folders
for word in WORDS:
    os.makedirs(os.path.join(DATA_PATH, word), exist_ok=True)

# Start capturing video
cap = cv2.VideoCapture(0)

for word in WORDS:
    print(f"Recording {word}. Press 'Q' to start.")
    while True:
        _, frame = cap.read()
        cv2.putText(frame, f"Press 'Q' to record {word}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for sequence in range(NUM_SEQUENCES):
        print(f"Recording {word}, Sequence {sequence + 1}/{NUM_SEQUENCES}")
        frames_data = []

        for frame_num in range(SEQUENCE_LENGTH):
            _, frame = cap.read()
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
                landmarks = [0] * 42  # 21 points * 2 (x, y)

            frames_data.append(landmarks)

            cv2.putText(frame,
                        f"Recording {word} {sequence + 1}/{NUM_SEQUENCES} Frame {frame_num + 1}/{SEQUENCE_LENGTH}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        np.save(os.path.join(DATA_PATH, word, f"{sequence}.npy"), np.array(frames_data))

    print(f"Finished recording {word}. Moving to the next word.")

cap.release()
cv2.destroyAllWindows()
