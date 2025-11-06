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
from PySide6.QtGui import QFontDatabase, QFont, QPixmap
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
COMMAND_GESTURES = {'arrival', 'description', 'traffic', 'voice', 'emergency', 'delete'}

# --- Helper Functions ---
def load_fonts():
    fonts_dir=ASSETS/"fonts";loaded={}
    def add_font(path,key):
        if path.exists():
            fid=QFontDatabase.addApplicationFont(str(path))
            fams=QFontDatabase.applicationFontFamilies(fid) if fid!=-1 else []
            if fams: loaded[key]=fams[0]
    add_font(fonts_dir/"SourceSans3-Regular.ttf","regular")
    add_font(fonts_dir/"SourceSans3-SemiBold.ttf","semibold")
    add_font(fonts_dir/"NotoSansKR-Regular.ttf","korean")
    return loaded

def start_web_server(host='localhost', port=5050, directory='.'):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.HTTPServer((host, port), handler)
    print(f"✅ Starting web server at http://{host}:{port}, serving from {directory}")
    httpd.serve_forever()

class GPSClient(QObject):
    new_location = Signal(float, float) # (lat, lng)

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
                    if "lat" in data and "lon" in data:
                        self.new_location.emit(data["lat"], data["lon"])
            except requests.RequestException:
                pass
            time.sleep(1)

    def stop(self):
        self._running = False
        print("🛑 GPS 클라이언트 중지.")

# =========================================================
# Main App Class
# =========================================================
class App(QWidget):
    def __init__(self, fonts: dict, screens):
        super().__init__()
        self.setWindowTitle("SignNav")
        self.resize(1024, 600)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.current_location = None
        self.hud_window = None
        self.TRANSITION_MS = 300
        self.TRANSITION_EASING = QEasingCurve.OutQuint

        # --- Setup HUD on secondary monitor if available ---
        if len(screens) > 1:
            main_screen_name = "HDMI-A-2"
            hud_screen_name  = "DSI-1"
            main_screen = hud_screen = None
            for screen in screens:
                print(f"감지된 모니터: {screen.name()}")
                if screen.name() == main_screen_name: main_screen = screen
                elif screen.name() == hud_screen_name: hud_screen = screen
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

        # --- Core Engines ---
        self.sign_engine = SignEngine(camera_type="webcam")  # "picam" 가능
        self.engine_thread = QThread()
        self.sign_engine.moveToThread(self.engine_thread)
        self.engine_thread.started.connect(self.sign_engine.initialize_and_run)

        self.gps_thread = QThread()
        self.gps_client = GPSClient()
        self.gps_client.moveToThread(self.gps_thread)
        self.gps_thread.started.connect(self.gps_client.run)

        # --- Root / Stacked ---
        root = QHBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # 전환 헬퍼
        def go(name, *, effect="slide-fade", direction="left", duration=None, easing=None):
            return lambda: self._switch(
                self.pages[name],
                effect=effect, direction=direction,
                duration=duration, easing=easing
            )

        self.pages = {}

        # --- Pages 생성 (상단바 콜백 통일) ---
        # 공통 콜백: 홈→Recognition, 보이스→Voice, 내비→Navigation, SOS→SOS
        cb_home = go("recognition", effect="fade", duration=300)
        cb_voice = go("voice", effect="fade", duration=300)
        cb_nav = go("navigation", effect="fade", duration=300)
        cb_sos = go("sos", effect="fade", duration=300)
        cb_recog = cb_home  # 일부 위젯에서 on_recog 이름을 쓰는 경우 대비

        # Welcome
        self.pages["welcome"] = WelcomePage(
            ASSETS,
            on_start=go("recognition", effect="fade", duration=300),
            fonts=fonts, sign_engine=self.sign_engine
        )

        # Recognition
        self.pages["recognition"] = RecognitionPage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_nav=cb_nav, on_sos=cb_sos,
            sign_engine=self.sign_engine
        )

        # Navigation
        self.navigation_page = NavigationPage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_nav=cb_nav, on_sos=cb_sos,
            sign_engine=self.sign_engine
        )
        self.pages["navigation"] = self.navigation_page

        # Description
        self.description_page = DescriptionPage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_nav=cb_nav, on_sos=cb_sos,
            fonts=fonts, sign_engine=self.sign_engine
        )
        self.pages["description"] = self.description_page

        # Search
        self.search_page = SearchPage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_nav=cb_nav, on_sos=cb_sos,
            fonts=fonts, sign_engine=self.sign_engine
        )
        self.pages["search"] = self.search_page
        if hasattr(self.search_page, 'set_main_app'):
            self.search_page.set_main_app(self)

        # SOS
        self.sos_page = SOSPage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_nav=cb_nav, on_sos=cb_sos,
            on_send=go("welcome", effect="fade", duration=300)
        )
        self.pages["sos"] = self.sos_page

        # Voice
        self.voice_page = VoicePage(
            ASSETS,
            on_home=cb_home, on_voice=cb_voice, on_recog=cb_recog, on_nav=cb_nav, on_sos=cb_sos,
            fonts=fonts, sign_engine=self.sign_engine
        )
        self.pages["voice"] = self.voice_page
        self.pages["voice"].playback_finished.connect(self._on_voice_playback_finished)

        # Stacked 등록 & 초기 페이지
        for p in self.pages.values():
            self.stack.addWidget(p)
        self.stack.setCurrentWidget(self.pages["welcome"])

        # --- Signals ---
        self.sign_engine.gesture_recognized.connect(self._handle_gesture)
        self.sign_engine.hangul_input_finished.connect(self._on_hangul_finished)
        self.sign_engine.session_finished.connect(self._on_session_finished)

        if self.hud_window:
            self.sign_engine.gesture_recognized.connect(self.hud_window.show_gesture_image)
            self.sign_engine.hangul_result_updated.connect(self.hud_window.update_text)
            self.sign_engine.hangul_input_finished.connect(lambda text: self.hud_window.update_text(text))

        self.gps_client.new_location.connect(self._update_location)
        self.stack.currentChanged.connect(self._on_page_changed)

        # sign→chat UIs
        try:
            self.sign_engine.hangul_input_finished.connect(self.pages["recognition"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["description"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["search"].append_user_text)
            self.sign_engine.hangul_result_updated.connect(self.voice_page._on_hangul_progress)
            self.sign_engine.hangul_input_finished.connect(self.voice_page.on_hangul_final)
            self.voice_page.playback_finished.connect(self._on_voice_playback_finished)
        except Exception:
            pass

        # Threads 시작
        self.engine_thread.start()
        self.gps_thread.start()
        print("✅ Sign Engine and GPS Client Threads started.")
        self._nav_pending = False

    # === Transition ===
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

        eff_from = QGraphicsOpacityEffect(from_lbl); from_lbl.setGraphicsEffect(eff_from); eff_from.setOpacity(1.0)
        eff_to   = QGraphicsOpacityEffect(to_lbl);   to_lbl.setGraphicsEffect(eff_to);     eff_to.setOpacity(0.0)

        grp = QParallelAnimationGroup(self)

        a_out = QPropertyAnimation(eff_from, b"opacity", self)
        a_out.setStartValue(1.0); a_out.setEndValue(0.0)
        a_out.setDuration(duration); a_out.setEasingCurve(QEasingCurve.InOutSine)
        grp.addAnimation(a_out)

        a_in  = QPropertyAnimation(eff_to, b"opacity", self)
        a_in.setStartValue(0.0); a_in.setEndValue(1.0)
        a_in.setDuration(duration); a_in.setEasingCurve(QEasingCurve.InOutSine)
        grp.addAnimation(a_in)

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

        def cleanup():
            try:
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
        QTimer.singleShot(duration + 200, lambda: (cleanup() if self._xfer_busy else None))
        grp.start()
        self._xfer_anim = grp  # GC 방지

    # --- 중앙 제어 ---
    @Slot(int)
    def _on_page_changed(self, index):
        current_page = self.stack.widget(index)
        page_name = type(current_page).__name__
        print(f"[App] Page changed to: {page_name}")

        if self.hud_window:
            self.hud_window.clear_text()
            if page_name == "RecognitionPage":
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

        else:  # Welcome, Recognition, SOS 등
            print("[App] Returning to Welcome Page.")
            self._switch(self.pages["welcome"], effect="fade")

    @Slot(float, float)
    def _update_location(self, lat, lng):
        self.current_location = (lat, lng)
        if hasattr(self, 'sos_page'):
            self.sos_page.set_location(lat, lng)
        if hasattr(self, 'navigation_page') and hasattr(self.navigation_page, 'set_location'):
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

# =========================================================
# HUD Window Class for Secondary Monitor
# =========================================================
class HUDWindow(QWidget):
    def __init__(self, fonts: dict):
        super().__init__()
        self.setWindowTitle("SignNav HUD")
        self.setStyleSheet("background-color: black;")

        self.layout = QStackedLayout(self)
        self.setLayout(self.layout)

        self.image_label = QLabel(); self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label  = QLabel(); self.text_label.setAlignment(Qt.AlignCenter)
        font = QFont(fonts.get("korean", "Arial"), 100, QFont.Bold)
        self.text_label.setFont(font)
        self.text_label.setStyleSheet("color: white;")
        self.blank_widget = QWidget()

        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.blank_widget)
        self.layout.setCurrentWidget(self.blank_widget)

        self.static_image_path = None
        self.temp_image_timer = QTimer(self)
        self.temp_image_timer.setSingleShot(True)
        self.temp_image_timer.timeout.connect(self._restore_static_image)

        self.gesture_images = {
            'arrival': ASSETS / "HUD" / "HUD_nav.png",
            'description': ASSETS / "HUD" / "HUD_discript.png",
            'traffic': ASSETS / "HUD" / "HUD_loc.png",
            'voice': ASSETS / "HUD" / "HUD_voice.png",
            'emergency': ASSETS / "HUD" / "HUD_sos.png",
            'delete': ASSETS / "HUD" / "HUD_warning.png"
        }

    def _set_pixmap(self, image_path):
        if image_path and image_path.exists():
            pixmap = QPixmap(str(image_path))
            self.image_label.setPixmap(pixmap.scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        else:
            self.image_label.clear()

    def _restore_static_image(self):
        if self.static_image_path:
            self._set_pixmap(self.static_image_path)
            self.layout.setCurrentWidget(self.image_label)
        else:
            self.layout.setCurrentWidget(self.blank_widget)

    def set_static_image(self, image_path):
        self.static_image_path = image_path
        self._restore_static_image()

    @Slot(str)
    def show_gesture_image(self, gesture: str):
        image_path = self.gesture_images.get(gesture)
        if image_path:
            self._set_pixmap(image_path)
            self.layout.setCurrentWidget(self.image_label)
            self.temp_image_timer.start(2000)

    @Slot(str)
    def update_text(self, text: str):
        self.text_label.setText(text)
        self.layout.setCurrentWidget(self.text_label)

    def clear_text(self):
        self.text_label.clear()
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
    base = QFont(fonts.get("regular", "Arial")); base.setPointSize(11)
    app.setFont(base)

    w = App(fonts, screens)
    w.showFullScreen()
    sys.exit(app.exec())
