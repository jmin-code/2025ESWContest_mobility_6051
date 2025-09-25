# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import threading
import http.server
import functools
import time

# --- Path Setup ---
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QHBoxLayout
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtCore import Slot, QThread, QTimer

DISABLE_GPS = os.getenv("DISABLE_GPS", "").lower() in {"1", "true", "yes", "on"}
GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyACM0")

if not DISABLE_GPS:
    from core.gps_reader import GPSReader
else:
    GPSReader = None  # 타입 힌트/가드용

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
COMMAND_GESTURES = {'arrival', 'description', 'traffic', 'voice'}

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

# =========================================================
# Main App Class
# =========================================================
class App(QWidget):
    def __init__(self, fonts: dict):
        super().__init__()
        self.setWindowTitle("SignNav")
        self.setFixedSize(1024, 600)
        self.current_location = None

        # --- 1. Core Engine/Reader Setup ---
        self.sign_engine = SignEngine()
        self.engine_thread = QThread()
        self.sign_engine.moveToThread(self.engine_thread)
        self.engine_thread.started.connect(self.sign_engine.initialize_and_run)

        # self.gps_reader = GPSReader(port='/dev/ttyACM0')
        # self.gps_thread = QThread()
        # self.gps_reader.moveToThread(self.gps_thread)
        # self.gps_thread.started.connect(self.gps_reader.run)
        
        # --- GPS (skippable via env) ---
        self.gps_thread = QThread()  # always create so closeEvent can safely check isRunning()
        if not DISABLE_GPS:
            self.gps_reader = GPSReader(port=GPS_PORT)
            self.gps_reader.moveToThread(self.gps_thread)
            self.gps_thread.started.connect(self.gps_reader.run)
        else:
            self.gps_reader = None
            print("[GPS] Disabled via DISABLE_GPS env; skipping serial port open.")
            # 선택: 개발 편의를 위해 대략적 현재 위치를 한 번 주입 (서울역)
            self._update_location(37.554678, 126.970609)
        
        # --- 2. UI Widget Setup ---
        root = QHBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # --- 3. Central Signal Connections ---
        self.sign_engine.gesture_recognized.connect(self._handle_gesture)
        self.sign_engine.hangul_input_finished.connect(self._on_hangul_finished)
        self.sign_engine.session_finished.connect(self._on_session_finished)
        
        # self.gps_reader.new_location.connect(self._update_location)
        if self.gps_reader:
            self.gps_reader.new_location.connect(self._update_location)
        
        self.stack.currentChanged.connect(self._on_page_changed)

        # --- 4. Start Background Threads ---
        self.engine_thread.start()
        # self.gps_thread.start()
        if self.gps_reader:
            self.gps_thread.start()
            
        print("✅ Sign Engine and GPS Threads started.")
        self._nav_pending = False  # arrival 처리 중복 방지

        # --- 5. Page Creation ---
        def go(name): return lambda: self.stack.setCurrentWidget(self.pages[name])
        
        self.pages = {}
        
        self.pages["welcome"] = WelcomePage(ASSETS, on_start=go("recognition"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["recognition"] = RecognitionPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_sos=go("sos"), sign_engine=self.sign_engine)
        self.navigation_page = NavigationPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["navigation"] = self.navigation_page        
        self.description_page = DescriptionPage(ASSETS, on_home=go("welcome"), on_recog=go("recognition"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["description"] = self.description_page
        self.search_page = SearchPage(ASSETS, on_home=go("welcome"), on_recog=go("recognition"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["search"] = self.search_page
        self.pages["sos"] = SOSPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_send=go("welcome"))
        self.pages["voice"] = VoicePage(ASSETS, fonts=fonts, sign_engine=self.sign_engine)
        
        for p in self.pages.values(): self.stack.addWidget(p)
        self.stack.setCurrentWidget(self.pages["welcome"])
        
        # --- Connect sign→text to chat UIs ---
        # 1) 제스처/한글 완료 → 인식 화면 채팅에 사용자 메시지로 추가
        # try:
        #     self.sign_engine.gesture_recognized.connect(self.pages["recognition"].append_user_text)
        # except Exception:
        #     pass
        try:
            self.sign_engine.hangul_input_finished.connect(self.pages["recognition"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["description"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["search"].append_user_text)
            self.sign_engine.hangul_result_updated.connect(self.pages["voice"]._on_hangul_progress)
            self.sign_engine.hangul_input_finished.connect(self.pages["voice"].on_hangul_final)


        except Exception:
            pass
        
        try:
            self.sign_engine.gesture_recognized.disconnect(self.pages["recognition"].append_user_text)
        except Exception:
            pass

    # --- Central Slot Methods ---
    @Slot(int)
    def _on_page_changed(self, index):
        current_page = self.stack.widget(index)
        print(f"[App] Page changed to: {type(current_page).__name__}")

        if isinstance(current_page, (WelcomePage, SOSPage)):
            self.sign_engine.switch_to_gesture_mode()
        elif isinstance(current_page, RecognitionPage):
            self._nav_pending = False 
            self.sign_engine.switch_to_gesture_mode()
            self.sign_engine.start_gesture_with_delay()
        elif isinstance(current_page, (NavigationPage, DescriptionPage, SearchPage, VoicePage)):
            self.sign_engine.start_hangul_with_delay()

    def _go_to_navigation(self):
        self.stack.setCurrentWidget(self.pages["navigation"])
        self._nav_pending = False

    def _go_to_description(self):
        self.stack.setCurrentWidget(self.pages["description"])

    def _go_to_search(self):
        self.stack.setCurrentWidget(self.pages["search"])
        
    def _go_to_voice(self):
        self.stack.setCurrentWidget(self.pages["voice"])
        

    @Slot(str)
    def _handle_gesture(self, gesture: str):
        current_page = self.stack.currentWidget()
        
        if isinstance(current_page, WelcomePage) and gesture == ACTIVATION_GESTURE:
            print(f"[App] Activation gesture '{gesture}' detected! Switching to Recognition Page.")
            self.stack.setCurrentWidget(self.pages["recognition"])
            
        # elif isinstance(current_page, RecognitionPage) and gesture in COMMAND_GESTURES:
        #     print(f"[App] Command gesture '{gesture}' detected!")
        #     if gesture == 'arrival':
        #         self.stack.setCurrentWidget(self.pages["navigation"])
        elif isinstance(current_page, RecognitionPage) and gesture in COMMAND_GESTURES:
            # arrival: 채팅(사용자→봇) 먼저, 2초 지연 전환
            if gesture == 'arrival':
                if self._nav_pending: return
                self._nav_pending = True
                page = self.pages["recognition"]
                try:
                    page.append_user_text("경로 설정")
                    QTimer.singleShot(100, lambda: page.append_bot_text("Navigation 화면으로 이동합니다!"))
                except Exception:
                    pass
                QTimer.singleShot(2000, self._go_to_navigation)
                return
            
            elif gesture == 'description':

                page = self.pages["recognition"]
                try:
                    page.append_user_text("정보 검색")
                    QTimer.singleShot(100, lambda: page.append_bot_text("정보 검색 화면으로 이동합니다!"))
                except Exception:
                    pass
                QTimer.singleShot(2000, self._go_to_description)
                return    
            
            elif gesture == 'traffic':

                page = self.pages["recognition"]
                try:
                    page.append_user_text("주변 인프라 탐색")
                    QTimer.singleShot(100, lambda: page.append_bot_text("주변 인프라 탐색 화면으로 이동합니다!"))
                except Exception:
                    pass
                QTimer.singleShot(2000, self._go_to_search)
                return    
            
            elif gesture == 'voice':
                page = self.pages["recognition"]
                try:
                    page.append_user_text("음성 안내")
                    QTimer.singleShot(100, lambda: page.append_bot_text("음성 안내를 시작합니다!"))
                except Exception:
                    pass
                QTimer.singleShot(2000, self._go_to_voice)                
                # 음성 안내 기능은 미구현 상태이므로 페이지 전환 없이 제스처 모드 유지
                return
        


    @Slot(str)
    def _on_hangul_finished(self, final_text: str):
        current_page = self.stack.currentWidget()
        if isinstance(current_page, NavigationPage):
            self.navigation_page.update_route(final_text)
        elif isinstance(current_page, DescriptionPage):
            self.description_page.search_for(final_text)
        elif isinstance(current_page, SearchPage):
            self.search_page.search_for(final_text)

    @Slot()
    def _on_session_finished(self):
        """'end' 제스처가 1초간 유지되어 세션을 종료/완료할 때 호출됨"""
        current_page = self.stack.currentWidget()
        page_name = type(current_page).__name__
        print(f"[App] Session finished signal received on {page_name}.")

        if isinstance(current_page, NavigationPage):
            final_text = self.sign_engine.get_hangul_result()
            print(f"[App] Hangul input finished with: '{final_text}'. Updating route.")

            self.navigation_page.update_route(final_text)
            self.sign_engine.switch_to_gesture_mode()

        elif isinstance(current_page, DescriptionPage):
            final_text = self.sign_engine.get_hangul_result()
            print(f"[App] Hangul input finished with: '{final_text}'. Searching for info.")
            self.description_page.search_for(final_text)
            self.sign_engine.switch_to_gesture_mode()

        elif isinstance(current_page, SearchPage):
            final_text = self.sign_engine.get_hangul_result()
            print(f"[App] Hangul input finished with: '{final_text}'. Searching for info.")
            self.search_page.search_for(final_text)
            self.sign_engine.switch_to_gesture_mode()
            
        elif isinstance(current_page, VoicePage):
            try:
                self.pages["voice"].on_end_gesture()
            except Exception as e:
                print("[App] VoicePage end gesture handling error:", e)

        else:
            print("[App] Returning to Welcome Page.")
            self.stack.setCurrentWidget(self.pages["welcome"])
        
    @Slot(float, float)
    def _update_location(self, lat, lng):
        self.current_location = (lat, lng)
        try:
            self.pages["sos"].set_location(lat, lng)
        except Exception:
            pass
    
    def closeEvent(self, event):
        print("Main window closing. Shutting down all threads.")
        # if hasattr(self, 'gps_reader'): self.gps_reader.stop()
        if getattr(self, 'gps_reader', None): self.gps_reader.stop()
        if hasattr(self, 'sign_engine'): self.sign_engine.stop()
        
        if self.gps_thread.isRunning(): self.gps_thread.quit(); self.gps_thread.wait()
        if self.engine_thread.isRunning(): self.engine_thread.quit(); self.engine_thread.wait()
        
        event.accept()

# --- Application Entry Point ---
if __name__ == "__main__":
    os.environ['QTWEBENGINE_REMOTE_DEBUGGING'] = "5051"
    app = QApplication(sys.argv)
    
    web_server_thread = threading.Thread(target=start_web_server, args=('localhost', 5050, ASSETS), daemon=True)
    web_server_thread.start()
    
    time.sleep(1)
    
    fonts = load_fonts()
    base = QFont(fonts.get("regular", "Arial"))
    base.setPointSize(11)
    app.setFont(base)
    
    w = App(fonts)
    w.show()
    sys.exit(app.exec())