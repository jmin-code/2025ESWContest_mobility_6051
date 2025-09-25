import cv2
import os

SAVE_DIR = "videos"
VIDEO_NAME = "gesture_조명.avi"
os.makedirs(SAVE_DIR, exist_ok=True)

# 웹캠 열기
cap = cv2.VideoCapture(0)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 프레임 가로
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 프레임 세로
fps = 20.0

# 비디오 코덱 설정 (XVID: .avi)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(SAVE_DIR, VIDEO_NAME), fourcc, fps, (width, height))

recording = False
print("[INFO] 스페이스바: 녹화 시작/정지 | q: 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if recording:
        out.write(frame)
        cv2.putText(frame, "REC", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow('Video Recorder', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 스페이스바: 녹화 토글
        recording = not recording
        print("[INFO] 녹화 시작" if recording else "[INFO] 녹화 정지")
    elif key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
