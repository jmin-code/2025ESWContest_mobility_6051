import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
import os
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image     #한글 폰트


# ============ 설정 ============
MODEL_PATH = "Model/hangul_chosung_model3.h5"
SCALER_PATH = "Model/scaler.joblib"

CONFIRMATION_THRESHOLD = 10     #동일 제스처가 몇 프레임 이상 유지되어야 확정될지
CONFIDENCE_THRESHOLD = 0.85     #모델 신뢰도 (softmax)기준
DISPLAY_TEXT_SIZE = 1.0         #텍스트 사이즈
COOLDOWN_TIME = 1               #제스처 확정 후 입력 대기 시간
ENABLE_CTC_LIKE = True          #같은 제스처 연속 확정 방지

HANGUL_LABELS = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ',
    10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ',
    31: 'ㄲ', 32: 'ㄸ', 33: 'ㅃ', 34: 'ㅆ', 35: 'ㅉ',
    36: 'ㄳ', 37: 'ㄵ', 38: 'ㄶ', 39: 'ㄺ', 40: 'ㄻ', 41: 'ㄼ', 42: 'ㄽ', 43: 'ㄾ', 44: 'ㄿ', 45: 'ㅀ',
    46: 'ㅄ',
    47: ' ', 48: 'backspace'
}

CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

#한글 그리기 함수
def draw_hangul_text(img, text, position, font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", font_size=32, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class HangulComposer:
    def __init__(self):
        self.stage = 0
        self.buffer = [None, None, None]
        self.pending_buffer = None  # 종성 보류용. 형식: [초, 중, 종]
        self.result_queue = deque()
        self.current_char_display = ""
        self.first_ever_input = True

    def combine_hangul(self, ch, ju, jo=None):
        try:
            cho_idx = CHOSUNG_LIST.index(ch)
            jung_idx = JUNGSUNG_LIST.index(ju)
            jong_idx = JONGSUNG_LIST.index(jo) if jo else 0
            return chr(0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx)
        except (ValueError, IndexError, TypeError):
            return ch if ch else None

    def update_current_char_display(self):
        if self.pending_buffer:
            result = self.combine_hangul(*self.pending_buffer)
            self.current_char_display = result if result else ""
            return

        cho, jung, jong = self.buffer
        if cho:
            result = self.combine_hangul(cho, jung, jong)
            self.current_char_display = result if result else ""
        else:
            self.current_char_display = ""

    def flush_pending_if_needed(self, next_label, frame_count=0, min_frames_for_cho=15):
        if not self.pending_buffer:
            return None

        pending_cho, pending_jung, pending_jong = self.pending_buffer

        if next_label in JUNGSUNG_LIST:
            char_without_jong = self.combine_hangul(pending_cho, pending_jung)
            if char_without_jong:
                self.result_queue.append(char_without_jong)
            self.buffer = [pending_jong, next_label, None]
            self.stage = 2
            self.pending_buffer = None
            return "[처리] 종성 재활용 → 새 글자 시작"

        elif next_label in CHOSUNG_LIST:
            if next_label == pending_jong:
                if frame_count < min_frames_for_cho:
                    return f"[무시] 같은 자음 '{next_label}' (프레임 {frame_count}<{min_frames_for_cho})"

            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char:
                self.result_queue.append(complete_char)
            self.buffer = [next_label, None, None]
            self.stage = 1
            self.pending_buffer = None
            return "[확정] 이전 글자 (종성 포함) → 새 초성 시작"

        else: # 공백, 백스페이스 등
            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char:
                self.result_queue.append(complete_char)
            self.pending_buffer = None
            self.buffer = [None, None, None]
            self.stage = 0
            return "[확정] 이전 글자 (종성 포함)"


    def flush_all_pending(self):
        if self.pending_buffer:
            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char:
                self.result_queue.append(complete_char)
            self.pending_buffer = None

        if self.buffer[0]:
            self.update_current_char_display()
            if self.current_char_display:
                self.result_queue.append(self.current_char_display)

        self.current_char_display = ""
        self.buffer = [None, None, None]
        self.stage = 0



    def process_input(self, label, frame_count=10):
            MIN_FRAMES_FOR_CHO = 80
            message = ""

            # [수정됨] 요청하신 4가지 복합 모음 규칙만 정의
            COMPOUND_JUNGSUNG = {
                'ㅗ': {'ㅏ': 'ㅘ', 'ㅐ': 'ㅙ'},
                'ㅜ': {'ㅓ': 'ㅝ', 'ㅔ': 'ㅞ'}
            }

            if self.first_ever_input and label not in CHOSUNG_LIST and label not in [' ', 'backspace']:
                return "[시스템] 첫 글자는 자음이어야 합니다."
            self.first_ever_input = False

            flush_msg = self.flush_pending_if_needed(label, frame_count=frame_count, min_frames_for_cho=MIN_FRAMES_FOR_CHO)
            if flush_msg:
                message += flush_msg
                if flush_msg.startswith("[무시]"):
                    self.update_current_char_display()
                    return message

                if "새 글자 시작" in flush_msg or "새 초성 시작" in flush_msg:
                    self.update_current_char_display()
                    return message

            if label == ' ':
                self.flush_all_pending()
                self.result_queue.append(' ')
                self.stage = 0
                return message + " | [시스템] 띄어쓰기"
            if label == 'backspace':
                if self.buffer[0] or self.pending_buffer:
                    self.buffer = [None, None, None]
                    self.pending_buffer = None
                    self.stage = 0
                    message += " | [시스템] 현재 글자 취소"
                elif self.result_queue:
                    self.result_queue.pop()
                    message += " | [시스템] 이전 글자 삭제"
                self.update_current_char_display()
                return message

            # ===== 모음 처리 로직 =====
            if label in JUNGSUNG_LIST:
                # stage가 2일 때 (초성+중성 상태에서 또 모음이 들어온 경우)
                if self.stage == 2:
                    prev_jung = self.buffer[1]
                    new_jung = label
                    
                    # 정의된 복합 모음 규칙에 맞는지 확인
                    if prev_jung in COMPOUND_JUNGSUNG and new_jung in COMPOUND_JUNGSUNG[prev_jung]:
                        # 규칙에 맞으면 중성을 복합 모음으로 교체
                        combined_jung = COMPOUND_JUNGSUNG[prev_jung][new_jung]
                        self.buffer[1] = combined_jung
                        message += f" | [결합] 모음 {prev_jung}+{new_jung} -> {combined_jung}"
                    else:
                        # 결합 불가능한 조합이면 무시
                        return f"[무시] '{prev_jung}' 뒤에 모음 '{new_jung}'는 올 수 없습니다."

                # stage가 1일 때 (초성만 있을 때) -> 정상적으로 중성을 결합
                elif self.stage == 1:
                    self.buffer[1] = label
                    self.stage = 2
                    message += " | [입력] 중성"
                # stage가 0 또는 다른 상태일 때 -> 'ㅇ'을 붙여서 새 글자를 생성
                else: 
                    self.flush_all_pending()
                    self.buffer = ['ㅇ', label, None]
                    self.stage = 2
                    message += " | [입력] 중성 ('ㅇ' 자동 추가)"

            # ===== 자음 처리 로직 (기존과 동일) =====
            elif label in CHOSUNG_LIST:
                if self.stage == 2: # 초성+중성 상태에서 자음이 들어오면 종성으로 처리
                    if label in JONGSUNG_LIST:
                        self.buffer[2] = label
                        self.pending_buffer = self.buffer.copy()
                        self.buffer = [None, None, None]
                        self.stage = 0
                        message += f" | [대기] 종성 '{label}' 후보"
                    else: 
                        self.flush_all_pending()
                        self.buffer = [label, None, None]
                        self.stage = 1
                        message += " | [처리] 이전 글자 확정, 새 초성 시작"
                else: # 그 외의 경우, 새 글자의 초성으로 처리
                    self.flush_all_pending()
                    self.buffer = [label, None, None]
                    self.stage = 1
                    message += " | [처리] 새 초성 시작"

            self.update_current_char_display()
            return message.strip(" |")

    def is_jongseong_candidate(self, label):
        """현재 입력된 label이 종성 후보와 일치하며, 결정을 기다리는 상태인지 확인"""
        if self.pending_buffer and self.pending_buffer[2]:
            return self.pending_buffer[2] == label
        return False


# ============ 모델 로딩 ============
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("[INFO] 모델 및 스케일러 로드 완료.")
except Exception as e:
    print(f"[ERROR] 모델 또는 스케일러 로드 실패: {e}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

composer = HangulComposer()
prev_id = None
same_count = 0
confirmed_id = None
cooldown_start_time = 0

print("[INFO] 실시간 한글 제스처 인식 시작 (종료: 'q' 키)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_text = "손 인식 대기..."
    text_color = (0, 255, 0)

    if time.time() - cooldown_start_time < COOLDOWN_TIME:
        display_text = "쿨다운 중..."
        text_color = (0, 165, 255)
        final_text = "".join(list(composer.result_queue))
        img = draw_hangul_text(img, f"Final: {final_text}", (10, 40), color=(0, 255, 0))
        img = draw_hangul_text(img, f"Current: {composer.current_char_display}", (10, 80), color=(0, 255, 255))
        #cv2.putText(img, display_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.imshow("Hangul Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        x_list = [lm.x for lm in hand_landmarks.landmark]
        y_list = [lm.y for lm in hand_landmarks.landmark]
        z_list = [lm.z for lm in hand_landmarks.landmark]
        landmarks_raw = np.array(x_list + y_list + z_list)

        if landmarks_raw.shape[0] == 63:
            input_data = np.expand_dims(landmarks_raw, axis=0)
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled, verbose=0)[0]
            class_id = np.argmax(prediction)
            confidence = prediction[class_id]
            current_prediction = HANGUL_LABELS.get(class_id, "?")
            display_text = f"예측: {current_prediction} (신뢰도: {confidence:.2f})"

            if confidence > CONFIDENCE_THRESHOLD:
                current_label = HANGUL_LABELS.get(class_id, "?")
                
                # CTC 무시 조건: 현재 제스처를 종성으로 기다릴 때는 CTC기능을 비활성화
                if ENABLE_CTC_LIKE and class_id == confirmed_id and not composer.is_jongseong_candidate(current_label):
                    pass # 일반적인 상황에서는 중복 입력 무시
                
                # 새로운 제스처거나, 종성 결정을 위해 프레임 카운트가 필요한 경우
                else:
                    if class_id == prev_id:
                        same_count += 1
                    else:
                        same_count = 1
                        # confirmed_id와 달라야만 prev_id 갱신 (중요)
                        prev_id = class_id

                    if same_count >= CONFIRMATION_THRESHOLD:
                        system_message = composer.process_input(current_label, frame_count=same_count)

                        # 작곡가가 "[무시]"가 아닌 유의미한 작업을 했을 때만 확정 처리
                        if not system_message.startswith("[무시]"):
                            confirmed_id = class_id
                            print(f"-> [확정] {current_label} (frames={same_count}) -> {system_message}")
                            cooldown_start_time = time.time()
                            # 확정 후에는 same_count를 초기화하여 연속적인 확정 방지
                            same_count = 0 
                        
                        text_color = (0, 255, 0) # 확정 시도 중
                    else:
                        text_color = (0, 165, 255) # 확정 대기 중
            else:
                # 신뢰도가 낮으면 모든 카운트 초기화
                same_count = 0
                prev_id = None
                text_color = (0, 255, 0)
        else:
            display_text = "랜드마크 오류"
            text_color = (0, 0, 255)

        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        same_count = 0
        prev_id = None
        confirmed_id = None

    final_text = "".join(list(composer.result_queue))
    img = draw_hangul_text(img, f"Final: {final_text}", (10, 40), font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", color=(0, 255, 0))
    img = draw_hangul_text(img, f"Current: {composer.current_char_display}", (10, 80), font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", color=(0, 255, 255))

    cv2.imshow("Hangul Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

composer.flush_all_pending()

final_text = ''.join(composer.result_queue)
print("[INFO] 최종 결과:", final_text)

print("[INFO] 종료됨.")
