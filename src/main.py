# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import threading
import http.server
import functools
import time
import requests

# --- Path Setup ---
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QHBoxLayout, QLabel, QVBoxLayout, QStackedLayout, QGraphicsOpacityEffect
from PySide6.QtGui import QFontDatabase, QFont, QPixmap, QTransform
from PySide6.QtCore import Slot, QThread, QTimer, Signal, QObject, Qt, QEasingCurve, QPropertyAnimation, QParallelAnimationGroup, QRect

DISABLE_GPS = os.getenv("DISABLE_GPS", "").lower() in {"1", "true", "yes", "on"}
GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyACM0")



# --- UI, Core, Asset Imports ---
from ui.welcome import WelcomePage
from ui.recognition import RecognitionPage
from ui.navigation import NavigationPage
from ui.sos import SOSPage
from ui.description import DescriptionPage
from ui.search import SearchPage
from ui.voice import VoicePage
from core.sign_engine import SignEngine

ASSETS = SRC_DIR / "ui" / "assets"

# --- Constants ---
ACTIVATION_GESTURE = 'start'
# 'delete' 제스처 추가
COMMAND_GESTURES = {'arrival', 'description', 'traffic', 'voice', 'emergency', 'delete'}
SPEED_THRESHOLD = 20.0 # 속도 제한 임계값 (km/h)
WARNING_IMAGE_PATH = ASSETS / "HUD" / "HUD_warning.png"


# --- Helper Functions ---
def load_fonts():
    fonts_dir=ASSETS/"fonts";loaded={}
    def add_font(path,key):
        if path.exists():
            fid=QFontDatabase.addApplicationFont(str(path))
            fams=QFontDatabase.applicationFontFamilies(fid) if fid!=-1 else []
            if fams: loaded[key]=fams[0]
    add_font(fonts_dir/"SourceSans3-Regular.ttf","regular");add_font(fonts_dir/"SourceSans3-SemiBold.ttf","semibold");add_font(fonts_dir/"NotoSansKR-Regular.ttf","korean");return loaded

def start_web_server(host='localhost', port=5050, directory='.'):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.HTTPServer((host, port), handler)
    print(f"✅ Starting web server at http://{host}:{port}, serving from {directory}")
    httpd.serve_forever()

class GPSClient(QObject):
    """
    백그라운드에서 gps.py 서버에 접속해 데이터를 가져오고,
    PySide6 UI 스레드로 안전하게 신호를 보내는 역할.
    """
    new_gps_data = Signal(float, float, float)  #(lat, lng, spd)

    def __init__(self, url="http://127.0.0.1:6051/api/gps"):
        super().__init__()
        self._url = url
        self._running = False

    def run(self):
        self._running = True
        print(f"✅ GPS 클라이언트 시작. 서버 polling: {self._url}")
        while self._running:
            try:
                response = requests.get(self._url, timeout=1)
                if response.status_code == 200:
                    data = response.json()
                    if "lat" in data and "lon" in data and "spd" in data:
                        self.new_gps_data.emit(data["lat"], data["lon"], data["spd"])
                # else:
                #     print(f"[GPSClient] 서버 응답 오류: {response.status_code}") # 디버깅용
            except requests.RequestException:
                # print("[GPSClient] 서버 연결 실패. gps.py가 실행 중인지 확인하세요.") # 디버깅용
                pass
            time.sleep(1) # 1초마다 데이터 요청

    def stop(self):
        self._running = False
        print("🛑 GPS 클라이언트 중지.")


# =========================================================
# Warning Page Class
# =========================================================
class WarningPage(QWidget):
    def __init__(self, image_path):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel()
        if Path(image_path).exists():
            pixmap = QPixmap(str(image_path))
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setAlignment(Qt.AlignCenter)
        else:
            label.setText("WARNING!")
            label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        self.setStyleSheet("background-color: black;")

        
# =========================================================
# Main App Class
# =========================================================
class App(QWidget):
    def __init__(self, fonts: dict, screens):
        super().__init__()
        self.setWindowTitle("SignNav")
        self.resize(1024, 600)
        self.current_location = None
        self.hud_window = None
        self.is_speeding = False
        self.TRANSITION_MS = 300
        self.TRANSITION_EASING = QEasingCurve.OutQuint

        if len(screens) > 1:
            main_screen_name = "HDMI-1-2"
            hud_screen_name = "DSI-1"

            main_screen = None
            hud_screen = None

            for screen in screens:
                print(f"감지된 모니터: {screen.name()}") # 디버깅을 위해 현재 모니터 이름 출력
                if screen.name() == main_screen_name:
                    main_screen = screen
                elif screen.name() == hud_screen_name:
                    hud_screen = screen

            if main_screen:
                self.move(main_screen.geometry().topLeft())
                print(f"✅ 메인 창을 {main_screen_name} 모니터로 이동했습니다.")

            if hud_screen:
                self.hud_window = HUDWindow(fonts)
                self.hud_window.setGeometry(hud_screen.geometry())
                self.hud_window.showFullScreen()
                print(f"✅ HUD 창을 {hud_screen_name} 모니터에 생성했습니다.")
            else:
                print(f"⚠️ HUD 모니터({hud_screen_name})를 찾을 수 없습니다.")
        else:
             print("⚠️ 모니터가 1대만 감지되었습니다. HUD를 생성하지 않습니다.")

        # --- 1. Core Engine/Reader Setup ---
        #self.sign_engine = SignEngine(camera_type= "picam")
        self.sign_engine = SignEngine(camera_type= "webcam")
        self.engine_thread = QThread()
        self.sign_engine.moveToThread(self.engine_thread)
        self.engine_thread.started.connect(self.sign_engine.initialize_and_run)

        # --- GPS (skippable via env) ---
        self.gps_thread = QThread()
        self.gps_client = GPSClient()
        self.gps_client.moveToThread(self.gps_thread)
        self.gps_thread.started.connect(self.gps_client.run)

        # --- 2. UI Widget Setup ---
        root = QHBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # --- 5. Page Creation ---
        # 전환 프리셋 헬퍼: effect / direction / duration / easing까지 한 번에
        def go(name, *, effect="slide-fade", direction="left",
            duration=None, easing=None):
            return lambda: self._switch(
                self.pages[name],
                effect=effect, direction=direction,
                duration=duration, easing=easing
            )

        self.pages = {}

        #과속 경고 페이지
        self.warning_page = WarningPage(WARNING_IMAGE_PATH)
        self.pages["warning"] = self.warning_page

        # Welcome
        self.pages["welcome"] = WelcomePage(
            ASSETS,
            on_start=go("recognition", effect="fade", duration=300),
            fonts=fonts, sign_engine=self.sign_engine
        )

        # Recognition
        self.pages["recognition"] = RecognitionPage(
            ASSETS,
            on_home=go("welcome", effect="fade", duration=300),
            on_voice=go("voice", effect="fade", duration=300),
            on_nav=go("navigation", effect="fade", duration=300),
            on_sos=go("sos", effect="fade", duration=300),
            sign_engine=self.sign_engine
        )

        # Navigation
        self.navigation_page = NavigationPage(
            ASSETS,
            on_home=go("welcome", effect="fade", duration=300),
            on_voice=go("voice", effect="fade", duration=300),
            on_nav=go("navigation", effect="fade", duration=300), 
            on_sos=go("sos", effect="fade", duration=300),
            sign_engine=self.sign_engine
        )
        self.pages["navigation"] = self.navigation_page

        # Description
        self.description_page = DescriptionPage(
            ASSETS,
            on_home=go("welcome", effect="fade", duration=300),
            on_recog=go("recognition", effect="fade", duration=300),  
            on_sos=go("sos", effect="fade", duration=300),
            fonts=fonts, sign_engine=self.sign_engine
        )
        self.pages["description"] = self.description_page

        # Search
        self.search_page = SearchPage(
            ASSETS,
            on_home=go("welcome", effect="fade", duration=300),
            on_recog=go("recognition", effect="fade", duration=300),
            on_sos=go("sos", effect="fade", duration=300),
            fonts=fonts, sign_engine=self.sign_engine
        )
        self.pages["search"] = self.search_page
        if hasattr(self.search_page, 'set_main_app'):
            self.search_page.set_main_app(self)

        # SOS
        self.sos_page = SOSPage(
            ASSETS,
            on_home=go("welcome", effect="fade", duration=300),
            on_voice=go("voice", effect="fade",  duration=500),
            on_nav=go("navigation", effect="fade", duration=300),
            on_send=go("welcome", effect="fade", duration=300)
        )
        self.pages["sos"] = self.sos_page

        # Voice
        self.voice_page = VoicePage(ASSETS, fonts=fonts, sign_engine=self.sign_engine)
        self.pages["voice"] = self.voice_page
        self.pages["voice"].playback_finished.connect(self._on_voice_playback_finished)

        # StackedWidget 등록 & 초기 페이지
        for p in self.pages.values():
            self.stack.addWidget(p)
        self.stack.setCurrentWidget(self.pages["welcome"])

        
        # --- 4. 중앙 시그널 연결 ---
        self.sign_engine.gesture_recognized.connect(self._handle_gesture)
        self.sign_engine.hangul_input_finished.connect(self._on_hangul_finished)
        self.sign_engine.session_finished.connect(self._on_session_finished)
        
        if self.hud_window:
            self.sign_engine.gesture_recognized.connect(self.hud_window.show_gesture_image)
            self.sign_engine.hangul_result_updated.connect(self.hud_window.update_text)
            self.sign_engine.hangul_input_finished.connect(lambda text: self.hud_window.update_text(text))
        
        # GPS 신호를 중앙 관리 슬롯에 연결
        self.gps_client.new_gps_data.connect(self._handle_gps_data)
        
        self.stack.currentChanged.connect(self._on_page_changed)

        # --- Connect sign→text to chat UIs ---
        try:
            self.sign_engine.hangul_input_finished.connect(self.pages["recognition"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["description"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["search"].append_user_text)
            self.sign_engine.hangul_result_updated.connect(self.voice_page._on_hangul_progress)
            self.sign_engine.hangul_input_finished.connect(self.voice_page.on_hangul_final)
            self.voice_page.playback_finished.connect(self._on_voice_playback_finished)
        except Exception:
            pass

        # --- 5. 백그라운드 스레드 시작 ---
        self.engine_thread.start()
        self.gps_thread.start()
        print("✅ Sign Engine and GPS Client Threads started.")
        self._nav_pending = False

        # === Smooth page transition (robust snapshot overlay) ===
    def _switch(self, to_widget: QWidget, *, effect: str = "fade",
                duration: int = None, easing: QEasingCurve.Type = None,
                direction: str = "left"):

        duration = duration if duration is not None else getattr(self, "TRANSITION_MS", 380)
        easing   = easing   if easing   is not None else getattr(self, "TRANSITION_EASING", QEasingCurve.InOutCubic)

        if not to_widget or self.stack.currentWidget() is to_widget:
            return
        if getattr(self, "_xfer_busy", False):
            QTimer.singleShot(30, lambda: self._switch(to_widget, effect=effect, duration=duration, easing=easing, direction=direction))
            return

        self._xfer_busy = True
        self._transitioning = True
        self.setCursor(Qt.WaitCursor)

        r = self.stack.rect(); w, h = r.width(), r.height()
        if w <= 0 or h <= 0:
            self.stack.setCurrentWidget(to_widget)
            self._xfer_busy = False
            self._transitioning = False
            self.unsetCursor()
            return

        frm = self.stack.currentWidget()

        if hasattr(self, "_xfer_overlay") and self._xfer_overlay:
            self._xfer_overlay.deleteLater()
        overlay = QWidget(self.stack)
        overlay.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        overlay.setGeometry(r); overlay.show()
        self._xfer_overlay = overlay

        pix_from = QPixmap(w, h)
        frm.render(pix_from)

        from_lbl = QLabel(overlay); from_lbl.setPixmap(pix_from)
        from_lbl.setGeometry(0, 0, w, h); from_lbl.show()

        to_lbl = QLabel(overlay)
        to_lbl.setStyleSheet("background:black;")
        to_lbl.setGeometry(0, 0, w, h); to_lbl.show()

        # 투명도 이펙트
        eff_from = QGraphicsOpacityEffect(from_lbl); from_lbl.setGraphicsEffect(eff_from); eff_from.setOpacity(1.0)
        eff_to   = QGraphicsOpacityEffect(to_lbl);   to_lbl.setGraphicsEffect(eff_to);     eff_to.setOpacity(0.0)

        grp = QParallelAnimationGroup(self)

        # 페이드
        a_out = QPropertyAnimation(eff_from, b"opacity", self)
        a_out.setStartValue(1.0); a_out.setEndValue(0.0)
        a_out.setDuration(duration); a_out.setEasingCurve(QEasingCurve.InOutSine)
        grp.addAnimation(a_out)

        a_in  = QPropertyAnimation(eff_to, b"opacity", self)
        a_in.setStartValue(0.0); a_in.setEndValue(1.0)
        a_in.setDuration(duration); a_in.setEasingCurve(QEasingCurve.InOutSine)
        grp.addAnimation(a_in)

        # (옵션) 미세 슬라이드: 효과가 slide-fade/slide일 때만 새 레이어를 12~32px 이동
        if effect in ("slide-fade", "slide"):
            def micro(delta): return max(12, min(32, delta // 30))
            dx = dy = 0
            d = (direction or "left").lower()
            if d in ("left", "←"):  dx =  micro(w)
            elif d in ("right", "→"): dx = -micro(w)
            elif d in ("up", "↑", "top"): dy =  micro(h)
            else: dy = -micro(h)
            start_geo = QRect(dx, dy, w, h); end_geo = QRect(0, 0, w, h)
            to_lbl.setGeometry(start_geo)
            a_slide = QPropertyAnimation(to_lbl, b"geometry", self)
            a_slide.setStartValue(start_geo); a_slide.setEndValue(end_geo)
            a_slide.setDuration(duration); a_slide.setEasingCurve(easing)
            grp.addAnimation(a_slide)

        # 반드시 cleanup 되도록 finally 보장
        def cleanup():
            try:
                # 실제 전환은 여기서 단 한 번
                self.stack.setCurrentWidget(to_widget)
            finally:
                try:
                    if self._xfer_overlay:
                        self._xfer_overlay.deleteLater()
                        self._xfer_overlay = None
                finally:
                    self._xfer_busy = False
                    self._transitioning = False
                    self.unsetCursor()

        grp.finished.connect(cleanup)

        # 혹시 모를 조기 종료/예외 대비 타임아웃 세이프티(애니가 finish 못할 때)
        QTimer.singleShot(duration + 200, lambda: (cleanup() if self._xfer_busy else None))

        grp.start()
        self._xfer_anim = grp  # GC 방지


    def _xfer_slide(self, frm: QWidget, to: QWidget, duration: int, easing, *, direction: str, with_fade: bool):
        r = self.stack.rect(); w, h = r.width(), r.height()
        dir_ = (direction or "left").lower()
        if dir_ in ("left", "←"):
            start_to, end_to = QRect(w, 0, w, h), QRect(0, 0, w, h)
            start_fr, end_fr = QRect(0, 0, w, h), QRect(-w//20, 0, w, h)  # 살짝 밀려나게
        elif dir_ in ("right", "→"):
            start_to, end_to = QRect(-w, 0, w, h), QRect(0, 0, w, h)
            start_fr, end_fr = QRect(0, 0, w, h), QRect(w//20, 0, w, h)
        elif dir_ in ("up", "↑", "top"):
            start_to, end_to = QRect(0, h, w, h), QRect(0, 0, w, h)
            start_fr, end_fr = QRect(0, 0, w, h), QRect(0, -h//20, w, h)
        else:  # down
            start_to, end_to = QRect(0, -h, w, h), QRect(0, 0, w, h)
            start_fr, end_fr = QRect(0, 0, w, h), QRect(0, h//20, w, h)

        grp = QParallelAnimationGroup(self)

        if frm:
            a_fr = QPropertyAnimation(frm, b"geometry", self)
            a_fr.setStartValue(start_fr); a_fr.setEndValue(end_fr)
            a_fr.setDuration(duration); a_fr.setEasingCurve(easing)
            grp.addAnimation(a_fr)

        to.setGeometry(start_to)
        a_to = QPropertyAnimation(to, b"geometry", self)
        a_to.setStartValue(start_to); a_to.setEndValue(end_to)
        a_to.setDuration(duration); a_to.setEasingCurve(easing)
        grp.addAnimation(a_to)

        eff_fr = eff_to = None
        if with_fade:
            eff_to = QGraphicsOpacityEffect(to); to.setGraphicsEffect(eff_to)
            a_op_to = QPropertyAnimation(eff_to, b"opacity", self)
            a_op_to.setStartValue(0.0); a_op_to.setEndValue(1.0)
            a_op_to.setDuration(duration); a_op_to.setEasingCurve(easing)
            grp.addAnimation(a_op_to)

            if frm:
                eff_fr = QGraphicsOpacityEffect(frm); frm.setGraphicsEffect(eff_fr)
                a_op_fr = QPropertyAnimation(eff_fr, b"opacity", self)
                a_op_fr.setStartValue(1.0); a_op_fr.setEndValue(0.0)
                a_op_fr.setDuration(duration); a_op_fr.setEasingCurve(easing)
                grp.addAnimation(a_op_fr)

        def cleanup():
            if eff_to: to.setGraphicsEffect(None)
            if eff_fr and frm: frm.setGraphicsEffect(None)
            # 지오메트리 복원
            to.setGeometry(self.stack.rect())
            if frm: frm.setGeometry(self.stack.rect())
            self._xfer_busy = False

        grp.finished.connect(cleanup)
        grp.start()
        self._xfer_anim = grp  # GC 방지


    # --- 중앙 제어 슬롯 메서드 ---
    @Slot(int)
    def _on_page_changed(self, index):
        current_page = self.stack.widget(index)
        page_name = type(current_page).__name__
        print(f"[App] Page changed to: {page_name}")

        if self.hud_window:
            self.hud_window.clear_text() # 페이지 변경 시 항상 텍스트는 초기화
            if page_name == "RecognitionPage":
                # Recognition 페이지일 경우 HUD_recog.png를 고정 이미지로 설정
                self.hud_window.set_static_image(ASSETS / "HUD" / "HUD_recog.png")
            else:
                self.hud_window.set_static_image(None)

        if page_name in ("WelcomePage", "SOSPage"):
            self.sign_engine.switch_to_gesture_mode()
        elif page_name == "RecognitionPage":
            self._nav_pending = False
            self.sign_engine.switch_to_gesture_mode()
            self.sign_engine.start_gesture_with_delay()
        else:
            self.sign_engine.start_hangul_with_delay()


    @Slot()
    def _on_voice_playback_finished(self):
        print("[App] Voice playback finished. Switching to gesture mode.")
        if self.sign_engine:
            self.sign_engine.switch_to_gesture_mode()

    def _go_to_navigation(self): self._switch(self.navigation_page, effect="fade")
    def _go_to_description(self): self._switch(self.description_page, effect="fade")
    def _go_to_search(self): self._switch(self.search_page, effect="fade")
    def _go_to_voice(self): self._switch(self.voice_page, effect="fade")
    def _go_to_sos(self): self._switch(self.sos_page, effect="fade")


    @Slot(str)
    def _handle_gesture(self, gesture: str):
        current_page = self.stack.currentWidget()
        page_name = type(current_page).__name__

        if gesture == 'delete':
            print(f"[App] 'delete' gesture recognized! Returning to Recognition Page.")
            self._switch(self.pages["recognition"], effect="fade")
            return

        if page_name == "WelcomePage" and gesture == ACTIVATION_GESTURE:
            print(f"[App] Activation gesture '{gesture}' detected! Switching to Recognition Page.")
            self._switch(self.pages["recognition"], effect="fade")
            
        elif page_name == "RecognitionPage" and gesture in COMMAND_GESTURES:
            page = self.pages["recognition"]
            
            if gesture == 'arrival':
                if self._nav_pending: return
                self._nav_pending = True
                page.append_user_text("경로 설정")
                QTimer.singleShot(100, lambda: page.append_bot_text("Navigation 화면으로 이동합니다!"))
                QTimer.singleShot(2000, self._go_to_navigation)
            
            elif gesture == 'description':
                page.append_user_text("정보 검색")
                QTimer.singleShot(100, lambda: page.append_bot_text("정보 검색 화면으로 이동합니다!"))
                QTimer.singleShot(2000, self._go_to_description)
            
            elif gesture == 'traffic':
                page.append_user_text("주변 인프라 탐색")
                QTimer.singleShot(100, lambda: page.append_bot_text("주변 인프라 탐색 화면으로 이동합니다!"))
                QTimer.singleShot(2000, self._go_to_search)
            
            elif gesture == 'voice':
                page.append_user_text("음성 안내")
                QTimer.singleShot(100, lambda: page.append_bot_text("음성 안내를 시작합니다!"))
                QTimer.singleShot(2000, self._go_to_voice)

            # <<< 추가: emergency 제스처로 SOS 페이지 이동
            elif gesture == 'emergency':
                page.append_user_text("긴급구조")
                QTimer.singleShot(100, lambda: page.append_bot_text("긴급구조 요청을 시작합니다!"))
                QTimer.singleShot(2000, self._go_to_sos)

    @Slot(str)
    def _on_hangul_finished(self, final_text: str):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, NavigationPage):
            self.navigation_page.update_route(final_text)
        elif isinstance(current_page, DescriptionPage):
            self.description_page.search_for(final_text)
        elif isinstance(current_page, SearchPage):
            self.search_page.search_for(final_text)

        if self.hud_window:
            QTimer.singleShot(3000, self.hud_window.clear_text)

    @Slot()
    def _on_session_finished(self):
        current_page = self.stack.currentWidget()
        page_name = type(current_page).__name__
        print(f"[App] Session finished signal received on {page_name}.")

        if page_name in ("NavigationPage", "DescriptionPage", "SearchPage"):
            final_text = self.sign_engine.get_hangul_result()
            print(f"[App] Hangul input finished with: '{final_text}'.")
            if page_name == "NavigationPage": self.navigation_page.update_route(final_text)
            elif page_name == "DescriptionPage": self.description_page.search_for(final_text)
            elif page_name == "SearchPage": self.search_page.search_for(final_text)
            self.sign_engine.switch_to_gesture_mode()
            
        elif page_name == "VoicePage":
            if hasattr(self.voice_page, 'on_end_gesture'):
                self.voice_page.on_end_gesture()

        else: # Welcome, Recognition, SOS 등
            print("[App] Returning to Welcome Page.")
            self._switch(self.pages["welcome"], effect="fade")
            
    @Slot(float, float)
    def _update_location(self, lat, lng):
        self.current_location = (lat, lng)
        if hasattr(self, 'sos_page'):
            self.sos_page.set_location(lat, lng)

        if hasattr(self, 'navigation_page'):
            if hasattr(self.navigation_page, 'set_location'):
                self.navigation_page.set_location(lat, lng)


    def closeEvent(self, event):
        print("Main window closing. Shutting down all threads.")
        if self.hud_window:
            self.hud_window.close()
        if hasattr(self, 'gps_client'): self.gps_client.stop()
        if hasattr(self, 'sign_engine'): self.sign_engine.stop()

        if self.gps_thread.isRunning(): self.gps_thread.quit(); self.gps_thread.wait()
        if self.engine_thread.isRunning(): self.engine_thread.quit(); self.engine_thread.wait()

        event.accept()


    @Slot(float, float, float)
    def _handle_gps_data(self, lat, lng, spd):
        self.current_location = (lat, lng)
        
        if hasattr(self, 'sos_page'): self.sos_page.set_location(lat, lng)
        if hasattr(self, 'navigation_page') and hasattr(self.navigation_page, 'set_location'):
            self.navigation_page.set_location(lat, lng)

        is_currently_speeding = spd > SPEED_THRESHOLD

        if is_currently_speeding and not self.is_speeding:
            print(f"⚠️ 과속 감지! 현재 속도: {spd:.1f} km/h. 경고 화면 표시.")
            self.is_speeding = True
            if hasattr(self.sign_engine, 'pause'): self.sign_engine.pause()
            
            self.stack.setCurrentWidget(self.warning_page)
            if self.hud_window: self.hud_window.set_static_image(WARNING_IMAGE_PATH)

        elif not is_currently_speeding and self.is_speeding:
            print(f"✅ 정상 속도 복귀. 현재 속도: {spd:.1f} km/h. UI 복원.")
            self.is_speeding = False
            if hasattr(self.sign_engine, 'resume'): self.sign_engine.resume()
            
            self._switch(self.pages["recognition"], effect="fade")
            if self.hud_window:
                self._on_page_changed(self.stack.indexOf(self.pages["recognition"]))


# =========================================================
# HUD Window Class for Secondary Monitor
# =========================================================
from PySide6.QtWidgets import QWidget, QLabel, QStackedLayout
from PySide6.QtGui import QPixmap, QFont, QTransform
from PySide6.QtCore import Qt, QTimer, Slot

class HUDWindow(QWidget):
    def __init__(self, fonts: dict):
        super().__init__()
        self.setWindowTitle("SignNav HUD")
        self.setStyleSheet("background-color: black;")

        self.layout = QStackedLayout(self)
        self.setLayout(self.layout)

        self.display_label = QLabel()
        self.display_label.setAlignment(Qt.AlignCenter)

        self._text_renderer_label = QLabel()
        self._text_renderer_label.setAlignment(Qt.AlignCenter)
        font = QFont(fonts.get("korean", "Arial"), 100, QFont.Bold)
        self._text_renderer_label.setFont(font)

        self._text_renderer_label.setStyleSheet("background-color: transparent; color: white;")
        
        self._text_renderer_label.setVisible(False)

        self.blank_widget = QWidget()
        
        self.layout.addWidget(self.display_label)
        self.layout.addWidget(self.blank_widget)
        self.layout.setCurrentWidget(self.blank_widget)

        self.static_image_path = None
        self.temp_image_timer = QTimer(self)
        self.temp_image_timer.setSingleShot(True)
        self.temp_image_timer.timeout.connect(self._restore_static_image)

        # ... (gesture_images 딕셔너리는 그대로) ...
        self.gesture_images = {
            'arrival': ASSETS / "HUD" / "HUD_nav.png",
            'description': ASSETS / "HUD" / "HUD_discript.png",
            'traffic': ASSETS / "HUD" / "HUD_loc.png",
            'voice': ASSETS / "HUD" / "HUD_voice.png",
            'emergency': ASSETS / "HUD" / "HUD_sos.png"
        }

    def _set_flipped_pixmap(self, pixmap: QPixmap):
        if pixmap.isNull():
            self.display_label.clear()
            return

        transform = QTransform().scale(-1, 1)
        flipped_pixmap = pixmap.transformed(transform)
        scaled_pixmap = flipped_pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.display_label.setPixmap(scaled_pixmap)

    def _restore_static_image(self):
        if self.static_image_path and self.static_image_path.exists():
            pixmap = QPixmap(str(self.static_image_path))
            self._set_flipped_pixmap(pixmap)
            self.layout.setCurrentWidget(self.display_label)
        else:
            self.layout.setCurrentWidget(self.blank_widget)

    def set_static_image(self, image_path):
        self.static_image_path = image_path
        self._restore_static_image()

    @Slot(str)
    def show_gesture_image(self, gesture: str):
        image_path = self.gesture_images.get(gesture)
        if image_path and image_path.exists():
            pixmap = QPixmap(str(image_path))
            self._set_flipped_pixmap(pixmap)
            self.layout.setCurrentWidget(self.display_label)
            self.temp_image_timer.start(2000)

    @Slot(str)
    def update_text(self, text: str):
        self._text_renderer_label.setText(text)
        
        self._text_renderer_label.setGeometry(self.rect())

        text_pixmap = QPixmap(self.size())
        text_pixmap.fill(Qt.transparent)
        self._text_renderer_label.render(text_pixmap)
        
        self._set_flipped_pixmap(text_pixmap)
        self.layout.setCurrentWidget(self.display_label)

    def clear_text(self):
        self._restore_static_image()

# --- Application Entry Point ---
if __name__ == "__main__":
    os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = "5051"
    app = QApplication(sys.argv)

    screens = app.screens()
    print(f"✅ Found {len(screens)} screens.")

    web_server_thread = threading.Thread(target=start_web_server, args=('localhost', 5050, ASSETS), daemon=True)
    web_server_thread.start()

    time.sleep(1)

    fonts = load_fonts()
    base = QFont(fonts.get("regular", "Arial"))
    base.setPointSize(11)
    app.setFont(base)

    w = App(fonts, screens)
    w.show()
    sys.exit(app.exec())