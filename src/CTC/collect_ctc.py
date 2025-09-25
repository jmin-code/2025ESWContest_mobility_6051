# src/CTC/collect_ctc.py
import cv2, os, numpy as np, mediapipe as mp
from datetime import datetime

GESTURES = ["arrival", "correct", "delete", "description", "emergency", "start", "traffic", "voice"]
GESTURE_NAME = "delete"  # 위 리스트 중 하나로 바꿔가며 수집

SAVE_DIR = f"dataset/{GESTURE_NAME}"
NUM_SAMPLES = 80
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
frames, recording = [], False
print(f"[INFO] '{GESTURE_NAME}' 수집 — SPACE 시작/종료, q 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)
        if recording:
            coords = [v for lm in hand.landmark for v in (lm.x, lm.y, lm.z)]
            frames.append(coords)
            if len(frames) > NUM_SAMPLES:
                frames = frames[-NUM_SAMPLES:]

    cv2.putText(img, ("Rec..." if recording else "SPACE to record"), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if recording else (0,0,255), 2)
    cv2.imshow("CTC Gesture Collector", img)
    k = cv2.waitKey(1)
    if k == ord(' '):
        if not recording:
            frames, recording = [], True
            print("[INFO] 녹화 시작")
        else:
            recording = False
            if len(frames) >= 5:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = os.path.join(SAVE_DIR, f"{GESTURE_NAME}_{ts}.npy")
                np.save(out, np.array(frames, dtype=np.float32))
                print(f"[SAVED] {out} ({len(frames)} frames)")
            else:
                print("[WARN] 프레임 부족, 저장 안 함")
            frames = []
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
