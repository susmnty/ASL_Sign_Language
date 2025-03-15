# ASL Sign Language
This project translates sign language gestures into text using a machine learning-based hand tracking system. By leveraging Mediapipe for hand landmark detection and Random Forest Classifier for gesture recognition, the system achieves high accuracy in real-time sign translation.

## Objective
The **ASL Sign Language Recognition System** bridges the communication gap by translating sign language gestures into readable text. It assists individuals with speech and hearing impairments by converting hand gestures into recognizable words with high accuracy.

---

## Project Overview

This project is divided into four main components for modularity and clarity:

1.**Data Collection (`data_collection.py`)** -
Captures and stores sign language gesture sequences.

2.**Data Preprocessing (`data_preprocessing.py`)** -
Normalizes and structures the dataset for model training.

3.**Model Training (`train_model.py`)** -
Processes data, trains a Random Forest Classifier, and evaluates performance.

4.**Real-Time Inference (`real_time_inference.py`)** -
Performs live sign language recognition and displays text predictions.

---

## Components

### 1. Data Collection
-**File Structure:**
Captures 50 sequences per word, each containing 50 frames.

-**Purpose:**
Stores structured data of hand landmarks for training.

---

### 2. Data Preprocessing
-**Purpose:**
Ensures uniform sequence length and normalizes hand landmark coordinates.

-**Key Features:**

 1.Pads shorter sequences to 50 frames.

 2.Trims longer sequences.

 3.Stores cleaned data in sign_data_cleaned.

-**Dependencies:**
Requires `Numpy` and `OpenCV` for processing.

---

### 3. Model Training
-**Workflow:**

1. Loads and processes sign language dataset.

2. Trains a Random Forest Classifier.

3. Saves the trained model as sign_model_rf.pkl.

-**Key Requirements:**

 - Python libraries: OpenCV, Mediapipe, Scikit-learn, Joblib.

 - Well-labeled hand gesture sequences.

-**Steps to Run:**

Place dataset in `./sign_data_cleaned` folder.

Execute script and select:

Option 1: Preprocess Data

Option 2: Train Classifier

Option 3: Real-Time Detection

Press `Esc` to exit detection.

---

### 4. Real-Time Inference
-**Purpose:**
Performs live sign recognition using a trained model.

-**Key Features:**

-**Model Loading**: Loads `sign_model_rf.pkl`.

-**Hand Tracking**: Detects 21 hand-knuckle landmarks.

-**Prediction**: Classifies gestures into words.

-**Visualization**: Displays real-time results.

-**Dependencies**:

-Libraries: OpenCV, Mediapipe, NumPy, Joblib.

-Hardware: Webcam for live detection.

-**Usage**:
Run the script and press "q" to quit.

---

## Challenges Faced
-Managing and integrating multiple project components.
-Ensuring accurate hand landmark detection.
-Overcoming model inaccuracies to achieve a final accuracy of 97.5%.

---

## Future Enhancements
-Expand the dataset with more sign language words.
-Improve accuracy using deep learning models (LSTMs, CNNs).
-Enhance real-time tracking for both hands.
-Integrate AI-powered predictive text suggestions.

---

## Hand-Landmark Example

### Sign Language Word Representation
![Image](https://github.com/user-attachments/assets/85ac9923-7337-4bc6-9e68-9aca721e832e)

### Mediapipe Hand Landmarks
![Mediapipe Hand Landmarks](https://mediapipe.dev/images/mobile/hand_landmarks.png)

---

## Additional Learning

-**OpenCV:** Library for computer vision and image processing.
[Learn more](https://www.youtube.com/watch?v=7irSQuL24qY) 

-**Mediapipe:** ML-based pipelines for hand tracking.
[Learn more](https://www.youtube.com/watch?v=VDCdWwldlx4) 

-**Hand Landmark Model:** Detects 21 key points for gesture recognition.
[More info](https://google.github.io/mediapipe/solutions/hands.html)

---

## References

- [Face Detection, Face Mesh, OpenPose, Holistic, Hand Detection Using Mediapipe](https://www.youtube.com/watch?v=VDCdWwldlx4)  
- [Introduction to OpenCV](https://www.youtube.com/watch?v=7irSQuL24qY)

---

### Folder Structure

```plaintext
Sign-Language-Recognition/
│
├── sign_data/                # Raw sign language dataset
├── sign_data_cleaned/        # Preprocessed dataset
├── models/                   # Trained model files (e.g., sign_model_rf.pkl)
├── data_collection.py        # Script for collecting gesture sequences
├── data_preprocessing.py     # Script for preprocessing dataset
├── train_model.py            # Script for training classifier
├── real_time_inference.py    # Live recognition script
├── README.md                 # Project documentation
