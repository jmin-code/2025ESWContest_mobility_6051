import cv2
import os
import numpy as np
import mediapipe as mp
import time
from datetime import datetime


# 1. 설정 및 초기화

# ID와 한글 초성 매핑 딕셔너리 : 모델 학습 시 ID를 클래스 라벨로 사용
HANGUL = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ',
    5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',                                                             
    # 모음 (Vowels) - 14번부터 시작                        
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ',
    19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ',  
    29: 'ㅟ', 30: 'ㅢ', 
    #마무리                        
    31: 'end',
    #쌍자음
    32: 'ㄲ', 33: 'ㄸ', 34: 'ㅃ', 35: 'ㅆ', 36: 'ㅉ',
    #띄어쓰기, 삭제
    47: ' ', 48: 'backspace'
}                                                                            
            
# ---------------------------------------                   
# ID 변경하여 현재 수집할 초성 선택 !!!                                                                                                                   
                                                                                                                                                                                                                                         
              
CURRENT_CHOSUNG_ID = 47                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    


CURRENT_CHOSUNG = HANGUL[CURRENT_CHOSUNG_ID]             
                     
# --- MISSING DEFINITION FIX START ---
# CURRENT_CHOSUNG_NAME을 여기서 정의해야 합니다.
try:
    CURRENT_CHOSUNG_NAME = HANGUL[CURRENT_CHOSUNG_ID]
except KeyError: 
    print(f"[ERROR] 설정된 CURRENT_CHOSUNG_ID ({CURRENT_CHOSUNG_ID})에 해당하는 초성이 HANGUL 딕셔너리에 없습니다.")
    print("HANGUL 딕셔너리를 확인하거나, CURRENT_CHOSUNG_ID를 올바르게 설정해주세요.")
    exit()

# NUM_SAMPLES_PER_CHOSUNG을 NUM_SAMPLES와 동일하게 정의하거나,  
# 아래 출력 및 저장 부분에서 NUM_SAMPLES를 사용하도록 통일해야 합니다.
# 여기서는 NUM_SAMPLES를 사용하도록 가정하고, 출력부를 수정합니다.
# 만약 NUM_SAMPLES_PER_CHOSUNG이라는 별도 변수를 사용하고 싶다면,
# NUM_SAMPLES를 이 변수로 변경하고 NUM_SAMPLES_PER_CHOSUNG = 50 등으로 명시적으로 정의하세요.                                              
# 현재 코드 흐름상 NUM_SAMPLES가 총 목표 샘플 개수를 의미합니다.
# --- MISSING DEFINITION FIX END ---
                 
SAVE_DIR = f"./hangul/{CURRENT_CHOSUNG_ID}"
            
NUM_SAMPLES = 1000   # 이 변수가 총 목표 샘플 개수를 의미합니다.           


             
os.makedirs(SAVE_DIR, exist_ok=True)                                                
                   
# MediaPipe 초기화              
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5                                  
)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)                                                                               
# --- WEBCAM CHECK FIX START ---
if not cap.isOpened():
    print("[ERROR] 웹캠을 열 수 없습니다. 카메라가 연결되어 있는지 확인하거나, 다른 카메라 ID를 시도해보세요.")
    exit()
# --- WEBCAM CHECK FIX END ---

print(f"[INFO] 현재 수집 초성: '{CURRENT_CHOSUNG_NAME}'")
print(f"[INFO] 목표 샘플 개수: {NUM_SAMPLES}개") # NUM_SAMPLES로 변경
print(f"[INFO] ** 스페이스바를 누를 때마다 현재 손 모양 데이터가 저장됩니다. **")
print(f"[INFO] 손 모양을 유지한 채 위치와 각도를 다양하게 바꿔가며 저장하세요.")
print(f"[INFO] 'q' 키를 누르면 언제든지 종료됩니다.")

# 2. 데이터 수집 루프
current_sample_count = 0 # 현재까지 수집된 샘플 개수
last_save_time = 0  # 마지막 저장 시간 (스페이스바 연타 방지용)
SAVE_INTERVAL = 0 # 최소 저장 간격 (초). 너무 빠르게 동일 데이터가 저장되는 것 방지

# FPS 측정 초기화
fps = 0
frame_count = 0
fps_timer = time.perf_counter()

while current_sample_count < NUM_SAMPLES: # NUM_SAMPLES로 변경
    # 루프 시작 시간 기록 (FPS 계산용)
    start_time_loop = time.perf_counter() 
    
    ret, frame = cap.read() # 웹캠에서 프레임 읽기
    if not ret:
        print("[ERROR] 프레임을 읽을 수 없습니다. 카메라 연결이 끊겼거나 오류가 발생했습니다.")
        break # 웹캠에서 더 이상 프레임을 읽을 수 없을 때 루프를 종료합니다.

    img = cv2.flip(frame, 1) # 좌우 반전 (거울 모드처럼 보이게)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MediaPipe 처리를 위해 BGR -> RGB 변환 (OpenCV 기본은 BGR)
    results = hands.process(rgb) # MediaPipe로 손 랜드마크 처리

    # 손이 감지되면 화면에 랜드마크 그리기
    if results.multi_hand_landmarks:
        # 감지된 첫 번째 손의 랜드마크 사용
        hand_landmarks = results.multi_hand_landmarks[0] 
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS) # 랜드마크와 연결선 그리기

    # 화면에 현재 진행 상황 텍스트 표시
    display_text = f"Current {current_sample_count}/{NUM_SAMPLES} (Press SpaceBar to Save)" # NUM_SAMPLES로 변경
    cv2.putText(img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # FPS 계산 및 화면에 표시
    frame_count += 1
    current_time_loop = time.perf_counter()
    if current_time_loop - fps_timer >= 1.0: # 1초마다 FPS 업데이트
        fps = frame_count / (current_time_loop - fps_timer)
        fps_timer = current_time_loop
        frame_count = 0
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Hangul Chosung Sign Recorder", img) # 화면에 웹캠 영상 표시

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF # 1ms 동안 키 입력 대기 (OS 호환성을 위해 0xFF 마스킹)
    
    if key == ord(' '):  # 스페이스바를 눌렀을 때
        # 손이 감지되었고, 마지막 저장으로부터 충분한 시간이 지났다면 저장 실행
        if results.multi_hand_landmarks and (current_time_loop - last_save_time) > SAVE_INTERVAL:
            # 랜드마크 좌표 추출 (x, y, z 순서로 1차원 배열로 평탄화)
            # MediaPipe는 기본적으로 x, y, z 순서로 랜드마크가 제공됩니다.
            coords = np.array([lm.x for lm in hand_landmarks.landmark] +
                              [lm.y for lm in hand_landmarks.landmark] +
                              [lm.z for lm in hand_landmarks.landmark])
            
            # 파일명에 초성 이름, ID, 타임스탬프를 포함하여 고유성 확보
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # 연월일_시분초_마이크로초
            filename = f"id{CURRENT_CHOSUNG_ID}_{timestamp}.npy"
            out_path = os.path.join(SAVE_DIR, filename)
            
            np.save(out_path, coords) # 랜드마크 데이터 NumPy 배열로 저장
            print(f"[INFO] 샘플 {current_sample_count+1}/{NUM_SAMPLES} 저장 완료: {filename}") # NUM_SAMPLES로 변경
            current_sample_count += 1 # 수집된 샘플 개수 증가
            last_save_time = current_time_loop # 마지막 저장 시간 업데이트
    
    elif key == ord('q'): # 'q' 키를 눌렀을 때 종료
        print("[INFO] 'q' 키를 눌러 수집을 중단합니다.")
        break

print(f"[INFO] '{CURRENT_CHOSUNG_NAME}' 초성 지문자 데이터 수집을 종료합니다. 총 {current_sample_count}개 샘플 저장됨.")
cap.release()          # 웹캠 자원 해제
cv2.destroyAllWindows() # 모든 OpenCV 창 닫기