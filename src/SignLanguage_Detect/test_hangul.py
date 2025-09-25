import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
import os

# === Settings ===
# SEQUENCE_LENGTH is not needed for static model testing.

# Model and Scaler file paths (must match paths where they were saved in train_model_hangul.py)
MODEL_PATH = "Model/hangul_chosung_model3.h5"
SCALER_PATH = "Model/scaler.joblib"

# Hangul phonetic mapping (for console output, not displayed on screen)
HANGUL_LABELS = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ',
    5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    # 모음 (Vowels) - 14번부터 시작
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ',
    19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ',  
    29: 'ㅟ', 30: 'ㅢ',
}         

# === Load Model and Scaler ===
try:
    model = load_model(MODEL_PATH)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("Please check the model file path and ensure the file is not corrupted.")
    exit()

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"[INFO] Scaler loaded: {SCALER_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load scaler: {e}")
    print("Please check the scaler file path or ensure the scaler was correctly saved by the training script.")
    exit()

# === Initialize MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7, # Keep consistent with training
    min_tracking_confidence=0.5   # Keep consistent with training (and data collection)
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
display_text = "Waiting for hand..." # Text to display on screen

print("[INFO] Starting Real-time Hangul Chosung Recognition (Press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera. Camera might be disconnected or an error occurred.")
        break

    img = cv2.flip(frame, 1) # Flip horizontally
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    results = hands.process(rgb) # Process hand landmarks with MediaPipe

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract landmark coordinates (flatten to a 1D array of x, y, z)
        coords = np.array([lm.x for lm in hand_landmarks.landmark] +
                          [lm.y for lm in hand_landmarks.landmark] +
                          [lm.z for lm in hand_landmarks.landmark])
        
        # Check if landmark count is 63 (for error prevention)
        if coords.shape[0] != 63:
            print(f"[WARNING] Extracted landmark count is {coords.shape[0]} (expected 63). Skipping prediction.")
            display_text = "Landmark Error"
        else:
            # Expand dimensions to match model input shape (1, 63)
            input_data = np.expand_dims(coords, axis=0)
            
            # Apply scaler (transform using the same scaler from training)
            input_data_scaled = scaler.transform(input_data)

            # Model prediction
            prediction_probs = model.predict(input_data_scaled, verbose=0)[0] # suppress prediction output
            
            # Get class ID with highest probability and its confidence
            class_id = np.argmax(prediction_probs)
            confidence = prediction_probs[class_id]

            # Set a confidence threshold (adjust as needed to filter uncertain predictions)
            CONFIDENCE_THRESHOLD = 0.7 # Adjust this value for stricter/looser filtering

            if confidence > CONFIDENCE_THRESHOLD:
                # Display ID on screen (English text)
                display_text = f"Predicted ID: {class_id} (Conf: {confidence:.2f})"
                # Print to console with Hangul label for debugging
                print(f"[Prediction] ID: {class_id} ({HANGUL_LABELS.get(class_id, 'Unknown')}) Conf: {confidence:.2f}")
            else:
                display_text = f"Uncertain... (Conf: {confidence:.2f})"


        # Draw landmarks on the image
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        # If no hand detected, reset prediction text
        display_text = "No Hand Detected"

    # Display prediction on screen
    cv2.putText(img, f"Prediction: {display_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) # Green color for text

    cv2.imshow("Hangul Chosung Recognition", img)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Real-time recognition ended.")