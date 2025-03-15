import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define constants
DATA_PATH = "sign_data"
WORDS = ["YES", "NO", "THANKYOU", "SORRY", "HELLO", "I LOVE YOU", "GOODBYE", "PLEASE", "YOU ARE WELCOME", "FAMILY", "HOUSE", "LOVE"]
SEQUENCE_LENGTH = 50
FEATURE_DIM = 42  # 21 landmarks * 2 (x, y)

X, y = [], []
for label, word in enumerate(WORDS):
    word_path = os.path.join(DATA_PATH, word)
    for file in os.listdir(word_path):
        file_path = os.path.join(word_path, file)
        try:
            array = np.load(file_path)
            if array.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
                print(f"Skipping {file}, incorrect shape {array.shape}")
                continue
            X.append(array.flatten())  # Flatten the sequence into a single feature vector
            y.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "sign_model_rf.pkl")
print("Model saved as sign_model_rf.pkl")
