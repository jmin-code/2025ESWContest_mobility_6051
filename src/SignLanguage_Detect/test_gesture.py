import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# === 설정 ===
SEQUENCE_LENGTH = 60
model = load_model("../Model/model.h5")
label_map = {
    0: "AC_off", 1: "AC_on", 2: "Arrival", 3: "Description", 4: "Emergency",
    5: "Light", 6: "Route", 7: "Temperature", 8: "Traffic",
    9: "Voice", 10: ""
}

# === Mediapipe 초기화 ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
sequence = []
predicted_label = ""
recording = False

print("[INFO] 제스처 인식 대기 중 (스페이스바 누르면 60프레임 인식 시작, q로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks and recording:
        landmarks = result.multi_hand_landmarks[0]
        coords = []
        for lm in landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])
        sequence.append(coords)

        mp_drawing.draw_landmarks(img, landmarks, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(sequence), axis=0)
            try:
                prediction = model.predict(input_data, verbose=0)
                class_id = int(np.argmax(prediction))
                predicted_label = label_map.get(class_id, "Unknown")
                print(f"[예측] {predicted_label}")
            except Exception as e:
                print(f"[오류] 예측 중 문제 발생: {e}")
            sequence = []
            recording = False

    # 텍스트 표시
    if recording:
        text = f"Recording... ({len(sequence)}/{SEQUENCE_LENGTH})"
        color = (0, 255, 0)
    else:
        text = "Press SPACE to record"
        color = (0, 0, 255)

    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(img, f"Predicted: {predicted_label}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Gesture Tester", img)

    key = cv2.waitKey(1)
    if key == ord(' '):
        if not recording:
            print("[INFO] 인식 시작: 제스처를 60프레임 동안 유지하세요.")
            sequence = []
            recording = True
            predicted_label = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
