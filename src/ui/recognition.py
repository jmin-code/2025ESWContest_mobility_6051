import sys
from PySide6.QtCore import Qt, QRect, QSize, Slot, QObject, Signal, QProcess, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from ui.chat import ChatPanel
from pathlib import Path 

class RecognitionPage(QWidget):
    BASE_W, BASE_H = 800, 480
    
    def __init__(self, assets_dir, on_home=None, on_nav=None, on_sos=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir
        self.on_home_cb = on_home or (lambda: None)
        self.on_nav, self.on_sos = on_nav or (lambda: None), on_sos or (lambda: None)
        self.sign_engine = sign_engine

        # --- UI 요소 생성 ---
        self.bg = QLabel(self)
        self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "recog_bg.png")

        self.camera_view = QLabel(self)
        self.camera_view.setStyleSheet("background-color: black;")
        # self.img_sign_bg = QLabel(self)
        # self.pm_sign_bg = self._load_pix(self.assets / "icons" / "sign_recognized_recog.png")
        
        # 상태 아이콘: 기본 ready, 관절 추출 중 recognizing, 제스처 확정 recognized
        self.img_status = QLabel(self)
        self.pm_ready       = self._load_pix(self.assets / "icons" / "ready.png")
        self.pm_recognizing = self._load_pix(self.assets / "icons" / "recognizing.png")
        self.pm_recog       = self._load_pix(self.assets / "icons" / "sign_recognized_recog.png")
        self._status_current = "ready"
        self._status_prev_nonblink = "ready"
        from PySide6.QtCore import QTimer
        self._status_timer = QTimer(self); self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self._set_status(self._status_prev_nonblink))
       
        self._just_shown_blocker = QTimer(self); self._just_shown_blocker.setSingleShot(True)
        self._is_just_shown = False
        
        #voice 중복 실행 방지용 쿨다운
        self._voice_cooldown = False
        self._voice_cd_timer = QTimer(self)
        self._voice_cd_timer.setSingleShot(True)
        self._voice_cd_timer.timeout.connect(lambda: setattr(self, "_voice_cooldown", False))
    
        self.lbl_gesture = QLabel("...", self)
        self.lbl_gesture.setFont(QFont("Arial", 12, QFont.Bold)); self.lbl_gesture.setAlignment(Qt.AlignCenter); self.lbl_gesture.setStyleSheet("color: #00BFFF; background: transparent;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont("Arial", 18, QFont.Bold)); self.lbl_hangul.setAlignment(Qt.AlignCenter); self.lbl_hangul.setStyleSheet("color: white; background: transparent;")

        # --- ChatPanel (sign → text chat) ---
        user_png = self.assets/"icons"/"user_bubble.png"
        bot_png  = self.assets/"icons"/"com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("recognitionChat")
        self.chat.hide()  # 초기엔 숨김; 레이아웃 후 보이기
        self._welcome_sent = False 

        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent"); b.setCursor(Qt.PointingHandCursor); b.clicked.connect(cb)
            return b
            
        self.btn_home = mk_icon("home_b.png", self._on_home_clicked)
        self.btn_nav  = mk_icon("nav.png", self.on_nav)
        self.btn_sos  = mk_icon("sos.png", self.on_sos)
        
        # --- 시그널 연결 ---
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)

            self.sign_engine.hangul_result_updated.connect(self.update_hangul_text)

            self.sign_engine.status_updated.connect(self._status_from_text)

            if hasattr(self.sign_engine, "gesture_recognized"):
                self.sign_engine.gesture_recognized.connect(self._on_gesture_recognized)
        
        self._relayout()
        
        self.camera_view.raise_()
        # self.img_sign_bg.raise_()
        self.img_status.raise_()
        self.lbl_gesture.raise_()
        self.lbl_hangul.raise_()
        self.btn_home.raise_()
        self.btn_nav.raise_()
        self.btn_sos.raise_()

    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W * 2 or pm.height() >= self.BASE_H * 2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0; w0, h0 = int(pm.width()/dpr), int(pm.height()/dpr); bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)): return box
        s = min(bw/w0, bh/h0); w, h = int(w0*s), int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    def _map_from_design(self, fit, x, y, w=None, h=None, *, right=None, bottom=None):
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H; X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None: W, H = int(round(w*sx)), int(round(h*sy))
        else: W, H = fit.width()-X+fit.x()-int(round((right or 0)*sx)), fit.height()-Y+fit.y()-int(round((bottom or 0)*sy))
        return QRect(X,Y,W,H)

    def _on_home_clicked(self):
        self.on_home_cb()
    
    def showEvent(self, event):
        super().showEvent(event)
        self._is_just_shown = True
        self._just_shown_blocker.start(100)

        if self.sign_engine:
            if hasattr(self.sign_engine, 'switch_to_gesture_mode'):
                 self.sign_engine.switch_to_gesture_mode()
                 
        # self.lbl_gesture.setText("...")
        self.lbl_hangul.setText("")
        self.lbl_gesture.hide()
        # self._relayout()
        # self.lbl_gesture.setText("...")
        self.lbl_hangul.setText("")
        self._relayout()
        self.chat.show(); self.chat.raise_()

        if not self._welcome_sent:
            self.chat.clear()
            self.append_bot_text("안녕하세요! 무엇을 도와드릴까요? 😊")
            self._welcome_sent = True
        #self._set_status("ready")
        

    def hideEvent(self, event):
        self._welcome_sent = False
        super().hideEvent(event)
        
        # [ADD] voice 창 실행 함수 (별도 프로세스, 메인 UI 블로킹 없음)
    def _launch_voice(self, text="안녕하세요. 음성 안내 모드가 실행됩니다.", caption="음성을 출력 중입니다..."):
        if self._voice_cooldown:
            return
        self._voice_cooldown = True
        self._voice_cd_timer.start(2500)  # 2.5초 쿨다운 (원하면 조정)

        # 1) 패키지 모듈로 실행: python -m ui.voice --say ... --caption ...
        ok = QProcess.startDetached(
            sys.executable,
            ["-m", "ui.voice", "--say", text, "--caption", caption]
        )
        if ok:
            return

        # 2) 실패 시 폴백: 직접 파일 경로로 실행
        voice_py = Path(__file__).resolve().parent / "voice.py"
        ok2 = QProcess.startDetached(
            sys.executable,
            [str(voice_py), "--say", text, "--caption", caption]
        )
        if not ok2:
            print("[ERROR] voice UI launch failed (module & file fallback)")

        
    @Slot(str)
    def _status_from_text(self, text: str):
        if self._is_just_shown and not self._just_shown_blocker.isActive():
             self._is_just_shown = False 
        if self._is_just_shown:
             return

        if not text: return
        t = text.strip().lower()
        
        # 'recognized' 점등 타이머가 도는 동안은 ready/recognizing 로의 변경을 무시
        if self._status_timer.isActive():
            # 인식 확정 신호가 또 오면 점등 시간만 갱신
            if any(k in t for k in ("recognized", "확인", "검출", "인식")):
                self._blink_recognized()
            return
        
        # 확정/검출
        if any(k in t for k in ("recognized", "확인", "검출", "인식")):
            self._blink_recognized(); return
        # 관절 추출 중(손 보임)
        if any(k in t for k in ("landmark", "tracking", "detected", "hand", "관절", "랜드마크", "추출")):
            self._set_status("recognizing"); return
        # 대기/안 보임
        if any(k in t for k in ("show gesture", "ready", "보여주세요", "대기")):
            self._set_status("ready"); return

    @Slot(object)
    def set_camera_image(self, frame: QImage):
        try:
            if frame.isNull() or self.camera_view.width() == 0: return
            scaled = QPixmap.fromImage(frame).scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled)
        except Exception as e:
            print(f"[RecogPage] set_camera_image error: {e}")
    
    @Slot(str)
    def update_hangul_text(self, text):
        self.lbl_hangul.setText(text)
    
    @Slot(str)
    def update_status_text(self, text):
        self.lbl_gesture.setText(text.upper())

    @Slot(str)
    def _on_gesture_recognized(self, gesture_name: str):
        if self._is_just_shown and not self._just_shown_blocker.isActive():
             self._is_just_shown = False
        if self._is_just_shown:
             return

        g = (gesture_name or "").lower().strip()

        if g == 'start':
            self._set_status("ready")
            return

        # (기존 동작 유지)
        self._blink_recognized()

    def _relayout(self):
        full = self.rect(); self.bg.setGeometry(full)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width()*dpr), int(fit.height()*dpr), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img); pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)
        else:
            self.bg.setStyleSheet("background-color: #000020;")

        self.camera_view.setGeometry(self._map_from_design(fit, 17, 81, w=478, h=341))
        
        # sign_bg_rect = self._map_from_design(fit, 328, 442, right=330, bottom=16)
        # self.img_sign_bg.setGeometry(sign_bg_rect)
        # if not self.pm_sign_bg.isNull():
        #     self.img_sign_bg.setPixmap(self.pm_sign_bg.scaled(sign_bg_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
        sign_bg_rect = self._map_from_design(fit, 328, 442, right=330, bottom=16)
        self.img_status.setGeometry(sign_bg_rect)
        self._apply_status_pixmap()
            
        
        gesture_rect = QRect(sign_bg_rect.x(), sign_bg_rect.y(), sign_bg_rect.width(), int(sign_bg_rect.height() * 0.5))
        hangul_rect = QRect(sign_bg_rect.x(), sign_bg_rect.y() + gesture_rect.height(), sign_bg_rect.width(), int(sign_bg_rect.height() * 0.5))
        self.lbl_gesture.setGeometry(gesture_rect)
        self.lbl_gesture.setVisible(False)  # 레이아웃 시에도 계속 숨김
        self.lbl_hangul.setGeometry(hangul_rect)
        
        # hatPanel을 '우측 패널'에 꽉 차게 배치
        right_panel = self._map_from_design(fit, 520, 79, w=260, h=341)  # (x,y,w,h) 디자인 값
        self.chat.setGeometry(right_panel)
        if not self.chat.isVisible(): self.chat.show()
        self.chat.raise_()

        # 채팅 UI 도입으로 기존 라벨은 보조 역할만 수행 (원하면 숨김)
        self.lbl_hangul.setVisible(False)

        for btn, x, y, w, h in (
            (self.btn_home, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(fit, x, y, w=w, h=h)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)
        
    @Slot(str)
    def append_user_text(self, text: str):
        if not text:
            return
        t = text.strip()
        if not t:
            return

        if t.lower() == "arrival":
            t = "경로 설정"
        elif t.lower() == "description":
            t = "목적지 정보 검색"
        elif t.lower() == "voice":
            t = "음성 안내 모드 실행"
            
        if getattr(self, "_last_user_msg", None) == t:
            return
        self._last_user_msg = t
        self.chat.append(t, role="user")

    @Slot(str)
    def append_bot_text(self, text: str):
        if not text: return
        t = text.strip()
        if getattr(self, "_last_bot_msg", None) == t:  # 중복 방지
            return
        self._last_bot_msg = t
        self.chat.append(t, role="bot")
            
    @Slot(str)
    def _bot_echo(self, s: str):
        t = (s or "").strip().lower()
        if any(k in t for k in ("안녕", "안녕하세요", "hi", "hello", "ㅎㅇ")):
            self.append_bot_text("안녕하세요! 무엇을 도와드릴까요? 😊"); return
        if any(k in t for k in ("안내", "경로", "가자", "출발")):
            self.append_bot_text("경로를 설정할게요!"); return
        if any(k in t for k in ("정보", "설명", "알려줘")):
            self.append_bot_text("네, 어느 목적지에 대한 정보가 필요하신가요?"); return
        if any(k in t for k in ("음성", "voice")):
            self.append_bot_text("음성 안내를 실행합니다."); return

        self.append_bot_text("네, 계속 입력해주세요.")

    def _status_pm(self):
        m = {"ready": self.pm_ready, "recognizing": self.pm_recognizing, "recognized": self.pm_recog}
        return m.get(self._status_current, self.pm_ready)

    def _apply_status_pixmap(self):
        pm = self._status_pm()
        if pm.isNull() or self.img_status.width() == 0: return
        self.img_status.setPixmap(pm.scaled(self.img_status.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_status(self, state: str):
        if state not in ("ready", "recognizing", "recognized"): return
        if state != "recognized":
            self._status_prev_nonblink = state
            self._status_timer.stop()
        self._status_current = state
        self._apply_status_pixmap()

    def _blink_recognized(self, ms: int = 2300):  # 2s 전환 + 여유
        self._set_status("recognized")
        self._status_timer.start(ms)
