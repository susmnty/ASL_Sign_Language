import numpy as np
import os

# Define paths
DATA_PATH = "sign_data"  # Original dataset
CLEANED_PATH = "sign_data_cleaned"  # Folder for fixed sequences

# Ensure the cleaned folder exists
os.makedirs(CLEANED_PATH, exist_ok=True)

# Constants
SEQUENCE_LENGTH = 50
FEATURE_DIM = 42  # 21 landmarks * 2 (x, y)

# Loop through each word class
for word in os.listdir(DATA_PATH):
    word_path = os.path.join(DATA_PATH, word)
    cleaned_word_path = os.path.join(CLEANED_PATH, word)
    os.makedirs(cleaned_word_path, exist_ok=True)

    for file in os.listdir(word_path):
        file_path = os.path.join(word_path, file)

        try:
            array = np.load(file_path)

            # Fix incorrect sequence lengths
            if array.shape[0] < SEQUENCE_LENGTH:
                # Pad with zeros if too short
                padding = np.zeros((SEQUENCE_LENGTH - array.shape[0], FEATURE_DIM))
                fixed_array = np.vstack((array, padding))
            elif array.shape[0] > SEQUENCE_LENGTH:
                # Trim if too long
                fixed_array = array[:SEQUENCE_LENGTH]
            else:
                fixed_array = array  # Keep as is if correct

            # Save the fixed sequence
            np.save(os.path.join(cleaned_word_path, file), fixed_array)

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("✅ Preprocessing completed! Fixed sequences saved in 'sign_data_cleaned'.")