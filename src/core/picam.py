import cv2
import numpy as np
import mediapipe as mp
import torch
import joblib
import time
import sys
import traceback
import json
from tensorflow.keras.models import load_model
from PySide6.QtCore import QObject, Signal,QTimer
from PySide6.QtGui import QImage
from pathlib import Path
from picamera2 import Picamera2

# --- Path and Module Setup ---
try:
    SRC_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR = SRC_DIR.parent
    sys.path.insert(0, str(SRC_DIR))
    from core.hangul_composer import HangulComposer, HANGUL_LABELS
    from CTC.lstm_classifier import LSTMGestureClassifier

    MODEL_DIR = BASE_DIR / 'Model'
    GESTURE_LSTM_CKPT = MODEL_DIR / 'gesture_lstm_cls.pth'
    GESTURE_LSTM_META = MODEL_DIR / 'gesture_lstm_cls.json'
    HANGUL_MODEL_PATH = MODEL_DIR / 'hangul_chosung_model3.h5'
    SCALER_PATH = MODEL_DIR / 'scaler.joblib'
    # *** 라벨 맵 경로 추가 ***
    HANGUL_LABEL_MAP_PATH = MODEL_DIR / 'hangul_label_map.json'

except Exception as e:
    print(f"FATAL: Engine setup failed: {e}")
    traceback.print_exc()

DEVICE = torch.device("cpu")
ACTIONS_REQUIRING_HANGUL = {'arrival', 'description', 'traffic', 'voice'}

class SignEngine(QObject):
    frame_updated = Signal(QImage)
    gesture_recognized = Signal(str)
    hangul_result_updated = Signal(str)
    status_updated = Signal(str)
    hangul_input_finished = Signal(str)
    session_finished = Signal()
    finished = Signal()
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        self.last_inference_time = 0
        self.INFERENCE_INTERVAL = 0.05   #100ms
        super().__init__(parent)
        self.last_mediapipe_results = None
        self.running = False
        self.mode = 'GESTURE'
        self.composer = HangulComposer()
        self.cap = None
        # *** 라벨 맵 변수 추가 ***
        self.idx2id = None

        self.pause_end_time = 0
        self.post_pause_mode = 'GESTURE'
        self.hand_was_present = False
        self.dynamic_gesture_frames = None
        self.DYNAMIC_FRAME_LENGTH = 20
        self.end_gesture_start_time = 0
        self.last_predicted_id = None
        self.consecutive_count = 0
        self.last_event_time = 0
        self.id_to_word = None
        self.fixed_len = 100
        self.CONFIRMATION_THRESHOLD = 5
        self.EVENT_COOLDOWN = 1.0
         
        self._last_status = None
        self._last_status_time = 0.0

    def _emit_status(self, text: str, min_interval: float = 0.3):
        now = time.time()
        if text != self._last_status or (now - self._last_status_time) > min_interval:
            self.status_updated.emit(text)
            self._last_status = text
            self._last_status_time = now

    def initialize_and_run(self):
        if not self._load_resources():
            self.status_updated.emit("Engine failed to initialize.")
            self.finished.emit()
            return
        
        # QTimer를 사용하여 _main_loop를 주기적으로 호출
        self.running = True
        self.timer = QTimer()
        self.timer.timeout.connect(self._process_single_frame) # 이름을 _process_single_frame으로 변경
        self.timer.start(33) # 약 30 FPS로 프레임 처리 시도

    def _load_resources(self):
        try:
            self.status_updated.emit("Checking model files...")

            # --- 모델 파일 체크 ---
            for p in [HANGUL_MODEL_PATH, SCALER_PATH, GESTURE_LSTM_CKPT,
                    GESTURE_LSTM_META, HANGUL_LABEL_MAP_PATH]:
                if not Path(p).exists():
                    raise FileNotFoundError(f"Cannot find model: {p}")
             
            # --- Mediapipe Hands 초기화 ---
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
             
            # --- 모델 로드 ---
            self.hangul_model = load_model(str(HANGUL_MODEL_PATH))
            self.scaler = joblib.load(SCALER_PATH)

            # *** 라벨 맵 로드 ***
            with open(HANGUL_LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
                self.idx2id = {int(k): int(v) for k, v in label_map['idx2id'].items()}
            print("[INFO] Hangul label map loaded.")
             
            with open(GESTURE_LSTM_META, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.id_to_word = {int(k): v for k, v in meta['id_to_word'].items()}
            self.fixed_len = int(meta.get('fixed_len', 100))

            self.gesture_model = LSTMGestureClassifier(
                input_dim=63,
                hidden=128,
                num_layers=2,
                num_classes=len(self.id_to_word),
                dropout=0.2
            )
            self.gesture_model.load_state_dict(
                torch.load(GESTURE_LSTM_CKPT, map_location=DEVICE, weights_only=True)
            )
            self.gesture_model.to(DEVICE)
            self.gesture_model.eval()
             
            # --- 카메라 초기화 (Picamera2 사용) ---
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)}))
            self.picam2.start()
             
            self.status_updated.emit("Ready.")
            return True

        except Exception as e:
            self.status_updated.emit(f"Initialization failed: {e}")
            traceback.print_exc()
            return False

    def _process_single_frame(self):
        if not self.running or not hasattr(self, "picam2"):
            return

        # 모드 전환 처리 (이전과 동일)
        if self.mode == 'PAUSED':
            if time.time() >= self.pause_end_time:
                if self.post_pause_mode == 'HANGUL':
                    self.switch_to_hangul_mode()
                else:
                    self.switch_to_gesture_mode()

        # --- 프레임 캡처 ---
        rgb_frame = self.picam2.capture_array("main")
        if rgb_frame is None:
            return
        rgb_frame = cv2.flip(rgb_frame, 1)

        # --- Mediapipe 추론 (주기적으로 실행) ---
        now = time.time()
        if self.mode != 'PAUSED':
            if (now - self.last_inference_time) >= self.INFERENCE_INTERVAL:
                # 추론을 실행하고, 그 결과를 self.last_mediapipe_results에 저장
                results = self.hands.process(rgb_frame)
                self.last_mediapipe_results = results # <<-- 중요! 결과를 클래스 변수에 저장
                self.process_frame(results)
                self.last_inference_time = now

        # --- 손 랜드마크 그리기 ---
        # 그리기용 BGR 복사본 생성
        bgr_display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # 매번 로컬 results가 아닌, 저장된 self.last_mediapipe_results를 확인
        if self.last_mediapipe_results and self.last_mediapipe_results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                bgr_display_frame,
                self.last_mediapipe_results.multi_hand_landmarks[0],
                self.mp_hands.HAND_CONNECTIONS
            )

        # --- Qt GUI에 프레임 전달 ---
        rgb_display_frame = cv2.cvtColor(bgr_display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_display_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.frame_updated.emit(qt_image)


    def stop(self):
        if self.running:
            self.running = False
            self.timer.stop()
            if hasattr(self, "picam2"):
                self.picam2.stop()
                print("==================Camera stopped======================")
            self.finished.emit()

    def process_frame(self, results):
        current_time = time.time()
        if (current_time - self.last_inference_time) < self.INFERENCE_INTERVAL:
            return
        self.last_inference_time = current_time
        hand_is_present = results.multi_hand_landmarks is not None
         
        if hand_is_present and not self.hand_was_present: self._emit_status("hand landmarks detected")
        elif not hand_is_present and self.hand_was_present: self._emit_status("ready")

        if not hand_is_present:
            self.hand_was_present = False; self.dynamic_gesture_frames = None; self.end_gesture_start_time = 0; self.last_predicted_id = None; self.consecutive_count = 0
            return

        self.hand_was_present = True
        if self.mode == 'IDLE': return

        lm_raw = np.array([lm.x for lm in results.multi_hand_landmarks[0].landmark] + [lm.y for lm in results.multi_hand_landmarks[0].landmark] + [lm.z for lm in results.multi_hand_landmarks[0].landmark])
        pred = self.hangul_model.predict(self.scaler.transform(np.expand_dims(lm_raw, axis=0)), verbose=0)[0]
         
        # 1. 모델이 예측한 '내부 인덱스' (새로운 번호표)를 찾음
        predicted_idx = np.argmax(pred)
        conf = pred[predicted_idx]
         
        # 2. '내부 인덱스'를 '원래 ID' (원래 번호)로 변환
        original_id = self.idx2id.get(predicted_idx)
         
        # 3. '원래 ID'를 사용해 실제 라벨 문자를 찾음
        pred_label = HANGUL_LABELS.get(original_id)

        if pred_label == 'end' and conf > 0.95:
            if self.end_gesture_start_time == 0: self.end_gesture_start_time = current_time
            elif current_time - self.end_gesture_start_time >= 1.0:
                self.session_finished.emit(); self.end_gesture_start_time = 0; return
        else: self.end_gesture_start_time = 0

        if self.mode == 'GESTURE':
            if self.dynamic_gesture_frames is None: self.dynamic_gesture_frames = []
            coords = [c for lm in results.multi_hand_landmarks[0].landmark for c in (lm.x, lm.y, lm.z)]
            self.dynamic_gesture_frames.append(coords)
            if len(self.dynamic_gesture_frames) >= self.DYNAMIC_FRAME_LENGTH:
                predicted = self.predict_gesture_ctc(np.array(self.dynamic_gesture_frames))
                self.gesture_recognized.emit(predicted); self._emit_status("recognized"); self.dynamic_gesture_frames = None
         
        elif self.mode == 'HANGUL':
            if conf > 0.85 and pred_label != 'end':
                # 연속성 체크는 '원래 ID'로 해야 함
                if original_id == self.last_predicted_id:
                    self.consecutive_count += 1
                else:
                    self.consecutive_count = 1
                self.last_predicted_id = original_id

                if self.consecutive_count >= self.CONFIRMATION_THRESHOLD and (current_time - self.last_event_time) > self.EVENT_COOLDOWN:
                    self.composer.process_input(pred_label, self.consecutive_count)
                    self.last_event_time = current_time
                    full_text = "".join(list(self.composer.result_queue)) + self.composer.get_current_char()
                    self.hangul_result_updated.emit(full_text)
            else:
                self.last_predicted_id = None
                self.consecutive_count = 0

    def start_hangul_with_delay(self):
        print("[Engine] Pausing for 1s, then will switch to HANGUL mode.")
        self.mode = 'PAUSED'
        self.pause_end_time = time.time() + 1.0
        self.post_pause_mode = 'HANGUL'
        self.status_updated.emit("잠시 대기 중...")

    def start_gesture_with_delay(self):
        print("[Engine] Pausing for 1s, then will switch to GESTURE mode.")
        self.mode = 'PAUSED'
        self.pause_end_time = time.time() + 1.0
        self.post_pause_mode = 'GESTURE'
        self.status_updated.emit("잠시 대기 중...")
         
    def switch_to_idle_mode(self):
        if self.mode != 'IDLE':
            print("[Engine] Switching to IDLE mode."); self.mode = 'IDLE'

    def get_hangul_result(self):
        self.composer.flush_all_pending()
        return "".join(self.composer.result_queue).strip()
     
    def switch_to_gesture_mode(self):
        if self.mode != 'GESTURE':
            print("[Engine] Switching to GESTURE mode.")
            self.mode = 'GESTURE'
            self.composer = HangulComposer()
            self.mode_changed.emit(self.mode)
            self._emit_status("제스처를 보여주세요")

    def switch_to_hangul_mode(self):
        if self.mode != 'HANGUL':
            print("[Engine] Switching to HANGUL mode.")
            self.mode = 'HANGUL'; self.composer.flush_all_pending()
            self.mode_changed.emit(self.mode); self._emit_status("한글 입력 대기 중...")

    def predict_gesture_ctc(self, sequence):
        try:
            if sequence.shape[0] < 8: return "Unknown"
            def _norm(s):
                s = s.reshape(-1, 21, 3); s -= s[:, 0:1, :]; n = np.linalg.norm(s[:, :, :2].max(1, keepdims=True) - s[:, :, :2].min(1, keepdims=True), axis=2, keepdims=True); n[n<1e-6]=1; s/=n; return s.reshape(-1, 63)
            def _resample(s, l): T,C=s.shape; src=np.linspace(0,T-1,T); dst=np.linspace(0,T-1,l); o=np.empty((l,C),np.float32); [o.__setitem__((slice(None),i),np.interp(dst,src,s[:,i])) for i in range(C)]; return o
            seq = _norm(sequence.astype(np.float32)); seq = _resample(seq, self.fixed_len)
            with torch.no_grad():
                logits = self.gesture_model(torch.from_numpy(seq).unsqueeze(0).to(DEVICE))
                return self.id_to_word.get(int(torch.argmax(logits, dim=-1).item()), "Unknown")
        except Exception as e:
            print(f"[predict] error: {e}"); return "Unknown"
        
        