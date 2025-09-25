import os
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import time
from PIL import ImageFont, ImageDraw, Image
import webbrowser
from urllib.parse import quote
import threading
import http.server
import functools
import sys

# =================================================================
# 1. 설정 및 모델 로딩
# =================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
from src.CTC.ctc_model import GestureCTCNet

GESTURE_MODEL_PATH = BASE_DIR / 'src' / 'CTC' / 'gesture_ctc_model.pth'
FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
DEVICE = torch.device("cpu")
HANGUL_MODEL_PATH = BASE_DIR / "Model" / "hangul_chosung_model3.h5"
SCALER_PATH = BASE_DIR / "Model" / "scaler.joblib"
HTML_DIR = BASE_DIR / 'src' / 'Kakao_API'


HANGUL_LABELS = {0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ', 8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ', 31: 'ㄲ', 32: 'ㄸ', 33: 'ㅃ', 34: 'ㅆ', 35: 'ㅉ', 36: 'ㄳ', 37: 'ㄵ', 38: 'ㄶ', 39: 'ㄺ', 40: 'ㄻ', 41: 'ㄼ', 42: 'ㄽ', 43: 'ㄾ', 44: 'ㄿ', 45: 'ㅀ', 46: 'ㅄ', 47: ' ', 48: 'backspace'}
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
GESTURE_LABEL_MAP = {0: 'arrival', 1: 'description', 2: 'emergency', 3: 'restaurant', 4: 'route', 5: 'temperature', 6: 'traffic', 7: 'voice'}
TRIGGER_GESTURES = {'arrival', 'description', 'voice'}


def handle_arrival_action(text):
    """'arrival' 제스처에 대한 액션: 카카오맵 길찾기 실행"""
    print(f"[ACTION] 'arrival' 동작 실행: 목적지 '{text}' 경로 탐색")
    if not text: return
    base_url = "http://localhost:5050/"
    url = f"{base_url}?to={quote(text)}&mode=car&auto=1"
    webbrowser.open(url)

def handle_voice_action(text):
    """'voice' 제스처에 대한 액션: 입력된 텍스트를 음성으로 출력 (TTS)"""
    print(f"[ACTION] 'voice' 동작 실행: '{text}'를 음성으로 출력합니다. (TTS 기능 추가 필요)")
    # 예: import pyttsx3; engine = pyttsx3.init(); engine.say(text); engine.runAndWait()

def handle_description_action(text):
    """'description' 제스처에 대한 액션: (기능 확장용)"""
    print(f"[ACTION] 'description' 동작 실행: 입력된 내용 - {text}")


ACTION_DISPATCHER = {
    'arrival': handle_arrival_action,
    'description': handle_description_action,
    'voice': handle_voice_action,
}


def start_web_server(host='localhost', port=5050, directory='.'):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.HTTPServer((host, port), handler)
    print(f"웹 서버 시작: http://{host}:{port}/ (Serving from: {directory})")
    httpd.serve_forever()

def draw_hangul_text(img, text, position, font_path=FONT_PATH, font_size=32, color=(0, 255, 0)):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        cv2.putText(img, "font not found", position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


class HangulComposer:
    def __init__(self):
        self.stage = 0; self.buffer = [None, None, None]; self.pending_buffer = None; self.result_queue = deque(); self.current_char_display = ""; self.first_ever_input = True
    
    
    def combine_hangul(self, ch, ju, jo=None):
        try:
            cho_idx = CHOSUNG_LIST.index(ch); jung_idx = JUNGSUNG_LIST.index(ju); jong_idx = JONGSUNG_LIST.index(jo) if jo else 0
            return chr(0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx)
        except (ValueError, IndexError, TypeError): return ch if ch else None


    def update_current_char_display(self):
        if self.pending_buffer: result = self.combine_hangul(*self.pending_buffer); self.current_char_display = result if result else ""; return
        cho, jung, jong = self.buffer
        if cho: result = self.combine_hangul(cho, jung, jong); self.current_char_display = result if result else ""
        else: self.current_char_display = ""


    def flush_all_pending(self):
        if self.pending_buffer:
            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char: self.result_queue.append(complete_char)
            self.pending_buffer = None
        if self.buffer[0]:
            self.update_current_char_display()
            if self.current_char_display: self.result_queue.append(self.current_char_display)
        self.current_char_display = ""; self.buffer = [None, None, None]; self.stage = 0


    def process_input(self, label, frame_count=10):
        MIN_FRAMES_FOR_CHO = 80; message = ""; COMPOUND_JUNGSUNG = {'ㅗ': {'ㅏ': 'ㅘ', 'ㅐ': 'ㅙ'},'ㅜ': {'ㅓ': 'ㅝ', 'ㅔ': 'ㅞ'}}
        if self.first_ever_input and label not in CHOSUNG_LIST and label not in [' ', 'backspace']: return "[시스템] 첫 글자는 자음이어야 합니다."
        self.first_ever_input = False
        flush_msg = self.flush_pending_if_needed(label, frame_count=frame_count, min_frames_for_cho=MIN_FRAMES_FOR_CHO)
        if flush_msg:
            message += flush_msg
            if flush_msg.startswith("[무시]"): self.update_current_char_display(); return message
            if "새 글자 시작" in flush_msg or "새 초성 시작" in flush_msg: self.update_current_char_display(); return message
        if label == ' ': self.flush_all_pending(); self.result_queue.append(' '); self.stage = 0; return message + " | [시스템] 띄어쓰기"
        if label == 'backspace':
            if self.buffer[0] or self.pending_buffer: self.buffer = [None, None, None]; self.pending_buffer = None; self.stage = 0; message += " | [시스템] 현재 글자 취소"
            elif self.result_queue: self.result_queue.pop(); message += " | [시스템] 이전 글자 삭제"
            self.update_current_char_display(); return message
        if label in JUNGSUNG_LIST:
            if self.stage == 2:
                prev_jung = self.buffer[1]; new_jung = label
                if prev_jung in COMPOUND_JUNGSUNG and new_jung in COMPOUND_JUNGSUNG[prev_jung]:
                    self.buffer[1] = COMPOUND_JUNGSUNG[prev_jung][new_jung]; message += f" | [결합] 모음 {prev_jung}+{new_jung} -> {self.buffer[1]}"
                else: return f"[무시] '{prev_jung}' 뒤에 모음 '{new_jung}'는 올 수 없습니다."
            elif self.stage == 1: self.buffer[1] = label; self.stage = 2; message += " | [입력] 중성"
            else: return "[무시] 모음을 입력하려면 자음이 먼저 와야 합니다."
        elif label in CHOSUNG_LIST:
            if self.stage == 2:
                if label in JONGSUNG_LIST: self.buffer[2] = label; self.pending_buffer = self.buffer.copy(); self.buffer = [None, None, None]; self.stage = 0; message += f" | [대기] 종성 '{label}' 후보"
                else: self.flush_all_pending(); self.buffer = [label, None, None]; self.stage = 1; message += " | [처리] 이전 글자 확정, 새 초성 시작"
            else: self.flush_all_pending(); self.buffer = [label, None, None]; self.stage = 1; message += " | [처리] 새 초성 시작"
        self.update_current_char_display(); return message.strip(" |")
    

    def flush_pending_if_needed(self, next_label, frame_count=0, min_frames_for_cho=15):
        if not self.pending_buffer: return None
        pending_cho, pending_jung, pending_jong = self.pending_buffer
        if next_label in JUNGSUNG_LIST:
            char_without_jong = self.combine_hangul(pending_cho, pending_jung);
            if char_without_jong: self.result_queue.append(char_without_jong)
            self.buffer = [pending_jong, next_label, None]; self.stage = 2; self.pending_buffer = None; return "[처리] 종성 재활용 → 새 글자 시작"
        elif next_label in CHOSUNG_LIST:
            if next_label == pending_jong and frame_count < min_frames_for_cho: return f"[무시] 같은 자음 '{next_label}' (프레임 {frame_count}<{min_frames_for_cho})"
            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char: self.result_queue.append(complete_char)
            self.buffer = [next_label, None, None]; self.stage = 1; self.pending_buffer = None; return "[확정] 이전 글자 (종성 포함) → 새 초성 시작"
        else:
            complete_char = self.combine_hangul(*self.pending_buffer)
            if complete_char: self.result_queue.append(complete_char)
            self.pending_buffer = None; self.buffer = [None, None, None]; self.stage = 0; return "[확정] 이전 글자 (종성 포함)"
        

    def is_jongseong_candidate(self, label):
        if self.pending_buffer and self.pending_buffer[2]: return self.pending_buffer[2] == label
        return False

# --- 메인 코드 시작 ---
if __name__ == "__main__":
    web_server_thread = threading.Thread(target=start_web_server, args=('localhost', 5050, HTML_DIR), daemon=True)
    web_server_thread.start()

    print("[INFO] 모델 및 라이브러리 로딩을 시작합니다...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    try:
        hangul_model = load_model(HANGUL_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        num_classes = len(GESTURE_LABEL_MAP)
        gesture_model = GestureCTCNet(input_dim=63, hidden_dim=128, num_classes=num_classes)
        gesture_model.load_state_dict(torch.load(GESTURE_MODEL_PATH, map_location=DEVICE))
        gesture_model.to(DEVICE)
        gesture_model.eval()
        print(f"[SUCCESS] 모든 모델 로드 완료.")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        exit()

    def predict_gesture_ctc(sequence):
        if len(sequence) == 0: return "Unknown"
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = gesture_model(input_tensor); pred = torch.argmax(out, dim=2).squeeze(1).cpu().numpy(); blank = len(GESTURE_LABEL_MAP); decoded = []
            prev = -1
            for p in pred:
                if p != prev and p != blank: decoded.append(p)
                prev = p
            return GESTURE_LABEL_MAP.get(decoded[0], "Unknown") if decoded else "Unknown"

    app_state = {
        'mode': 'GESTURE', 'recording': False, 'gesture_frames': [], 'result_text': "",
        'composer': HangulComposer(), 'last_trigger': None,'hangul_prev_id': None, 'hangul_same_count': 0,
        'hangul_confirmed_id': None, 'cooldown_start_time': 0, 'hangul_last_message': "",
        'transition_start_time': 0
    }

    cap = cv2.VideoCapture(0)
    print("\n[INFO] 애플리케이션 시작. 'q': 종료, 'space': 녹화, 'esc': 모드전환")
    print(f"[MODE] 현재 모드: {app_state['mode']}")

    while True:
        ret, frame = cap.read()
        if not ret: break

        img = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if app_state['mode'] == 'TRANSITION':
            img = draw_hangul_text(img, "잠시 후 한글 모드로 전환됩니다...", (50, 240), font_size=30, color=(0, 255, 255))
            if time.time() - app_state.get('transition_start_time', 0) >= 1.0:
                print("[INFO] 1초 경과. 한글 모드를 활성화합니다.")
                app_state['mode'] = 'HANGUL'; app_state['hangul_prev_id'] = None; app_state['hangul_confirmed_id'] = None; app_state['hangul_same_count'] = 0
        
        elif app_state['mode'] == 'GESTURE':
            status_text = "Recording..." if app_state['recording'] else "Press SPACE to Record Gesture"; color = (0, 255, 0) if app_state['recording'] else (0, 0, 255)
            cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            if app_state['result_text']: img = draw_hangul_text(img, f"Gesture: {app_state['result_text']}", (10, 70), font_size=30, color=(255, 255, 0))
            if results.multi_hand_landmarks and app_state['recording']:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y, lm.z)]
                app_state['gesture_frames'].append(coords)

        elif app_state['mode'] == 'HANGUL':
            display_text = "Waiting for hand..."
            if time.time() - app_state['cooldown_start_time'] < 1.0: display_text = "Cooldown..."
            elif results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_raw = np.array([lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark] + [lm.z for lm in hand_landmarks.landmark])
                input_scaled = scaler.transform(np.expand_dims(landmarks_raw, axis=0))
                prediction = hangul_model.predict(input_scaled, verbose=0)[0]
                class_id = np.argmax(prediction); confidence = prediction[class_id]
                current_prediction = HANGUL_LABELS.get(class_id, "?"); display_text = f"Predict: {current_prediction} ({confidence:.2f})"
                if confidence > 0.85:
                    if class_id == app_state['hangul_prev_id']: app_state['hangul_same_count'] += 1
                    else: app_state['hangul_same_count'] = 1
                    app_state['hangul_prev_id'] = class_id
                    if app_state['hangul_same_count'] >= 10 and class_id != app_state['hangul_confirmed_id']:
                        system_message = app_state['composer'].process_input(current_prediction)
                        if not system_message.startswith("[무시]"):
                            app_state['hangul_confirmed_id'] = class_id; app_state['cooldown_start_time'] = time.time()
                            app_state['hangul_last_message'] = f"[OK] {current_prediction} -> {system_message}"; print(app_state['hangul_last_message'])
                else: app_state['hangul_same_count'] = 0; app_state['hangul_prev_id'] = None
            else: app_state['hangul_confirmed_id'] = None
            final_text = "".join(list(app_state['composer'].result_queue)); img = draw_hangul_text(img, f"결과: {final_text}", (10, 40), font_size=30, color=(0, 255, 0))
            img = draw_hangul_text(img, f"입력중: {app_state['composer'].current_char_display}", (10, 80), font_size=30, color=(0, 255, 255)); img = draw_hangul_text(img, display_text, (10, 440), font_size=20, color=(255, 255, 0))
            img = draw_hangul_text(img, "모드: 한글 (ESC로 복귀)", (10, 470), font_size=20, color=(0, 255, 0))

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Integrated Gesture Recognition", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

        elif key == 27:  # ESC
            if app_state['mode'] == 'HANGUL':
                print("\n[MODE] 제스처 인식 모드로 전환합니다.")
                app_state['mode'] = 'GESTURE'
                app_state['composer'].flush_all_pending()
                final_text = ''.join(app_state['composer'].result_queue).strip()
                print(f"[INFO] 한글 입력 최종 결과: {final_text}")

                # 딕셔너리를 사용하여 적절한 함수를 찾아 실행
                trigger = app_state.get('last_trigger')
                if trigger in ACTION_DISPATCHER and final_text:
                    action_function = ACTION_DISPATCHER[trigger]
                    action_function(final_text)

                app_state['composer'] = HangulComposer()
                app_state['hangul_last_message'] = ""
                app_state['result_text'] = ""
                app_state['last_trigger'] = None

        elif key == ord(' '):
            if app_state['mode'] == 'GESTURE':
                if not app_state['recording']:
                    app_state['recording'] = True; app_state['gesture_frames'] = []; app_state['result_text'] = ""; print("[INFO] 제스처 녹화를 시작합니다.")
                else:
                    app_state['recording'] = False
                    print(f"[INFO] 녹화 종료. {len(app_state['gesture_frames'])} 프레임 녹화됨.")
                    if len(app_state['gesture_frames']) > 10:
                        predicted_gesture = predict_gesture_ctc(np.array(app_state['gesture_frames']))
                        app_state['result_text'] = predicted_gesture
                        print(f"[RESULT] 예측된 제스처: {predicted_gesture}")
                        if predicted_gesture in TRIGGER_GESTURES:
                            print(f"[MODE] 트리거 제스처 '{predicted_gesture}' 인식! 1초 후 한글 모드로 전환합니다.")
                            app_state['mode'] = 'TRANSITION'
                            app_state['transition_start_time'] = time.time()
                            app_state['last_trigger'] = predicted_gesture
                    else:
                        app_state['result_text'] = "Too short"

    cap.release()
    if app_state['composer'].result_queue or app_state['composer'].current_char_display:
        app_state['composer'].flush_all_pending()
        final_text = ''.join(app_state['composer'].result_queue)
        print(f"\n[INFO] 최종 결과물: {final_text}")

    cv2.destroyAllWindows()
    print("[INFO] 애플리케이션이 종료되었습니다.")