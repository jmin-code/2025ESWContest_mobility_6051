import cv2
import mediapipe as mp
from picamera2 import Picamera2
import time

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Picamera2 초기화
picam2 = Picamera2()
# ✅ 4채널(XBGR8888) 대신 3채널(RGB888)을 사용하도록 명시적으로 설정합니다.
picam2.configure(picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

print("✅ 카메라와 MediaPipe 초기화 완료. 'q' 키를 누르면 종료됩니다.")

# FPS 계산을 위한 변수
prev_time = 0

try:
    while True:
        # 카메라에서 프레임 캡처 (이제 RGB 형식입니다)
        rgb_frame = picam2.capture_array()
        
        # 좌우 반전
        rgb_frame = cv2.flip(rgb_frame, 1)

        # 성능 향상을 위해 이미지를 읽기 전용으로 설정
        rgb_frame.flags.writeable = False
        
        # 손 랜드마크 감지
        results = hands.process(rgb_frame)
        
        # 이미지를 다시 쓰기 가능으로 변경 (랜드마크를 그리기 위함)
        rgb_frame.flags.writeable = True

        # 감지된 랜드마크를 원본 프레임에 그리기
        if results.multi_hand_landmarks:
            print("🖐️ 손 감지됨!")
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    rgb_frame, # 이제 이 프레임에 바로 그립니다.
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
        
        # FPS 계산 및 표시
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(rgb_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 화면에 결과 표시 (OpenCV는 BGR을 기대하므로 다시 변환)
        display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('MediaPipe Hands Test', display_frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 자원 해제
    print("\n프로그램을 종료합니다.")
    hands.close()
    picam2.stop()
    cv2.destroyAllWindows()