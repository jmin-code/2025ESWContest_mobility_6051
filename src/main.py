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

from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QHBoxLayout
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtCore import Slot, QThread, QTimer,Signal, QObject

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

class GPSClient(QObject):
    """
    백그라운드에서 gps.py 서버에 접속해 데이터를 가져오고,
    PySide6 UI 스레드로 안전하게 신호를 보내는 역할.
    """
    new_location = Signal(float, float) # (lat, lng) 신호 정의

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
# Main App Class
# =========================================================
class App(QWidget):
    def __init__(self, fonts: dict):
        super().__init__()
        self.setWindowTitle("SignNav")
        self.setFixedSize(1024, 600)
        self.current_location = None

        # --- 1. Core Engine / Reader 설정 ---
        # SignEngine 설정
        self.sign_engine = SignEngine()
        self.engine_thread = QThread()
        self.sign_engine.moveToThread(self.engine_thread)
        self.engine_thread.started.connect(self.sign_engine.initialize_and_run)

        # GPSClient 설정 (기존 GPSReader 관련 코드는 모두 정리)
        self.gps_thread = QThread()
        self.gps_client = GPSClient()
        self.gps_client.moveToThread(self.gps_thread)
        self.gps_thread.started.connect(self.gps_client.run)
        
        # --- 2. UI 위젯 설정 ---
        root = QHBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        # --- 3. 페이지 생성 ---
        # 페이지 전환을 위한 람다 함수
        def go(name): return lambda: self.stack.setCurrentWidget(self.pages[name])
        
        self.pages = {}
        
        self.pages["welcome"] = WelcomePage(ASSETS, on_start=go("recognition"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["recognition"] = RecognitionPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_sos=go("sos"), sign_engine=self.sign_engine)
        
        # 각 페이지를 인스턴스 변수로 저장하여 쉽게 접근하도록 변경
        self.navigation_page = NavigationPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["navigation"] = self.navigation_page
        
        self.description_page = DescriptionPage(ASSETS, on_home=go("welcome"), on_recog=go("recognition"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["description"] = self.description_page
        
        self.search_page = SearchPage(ASSETS, on_home=go("welcome"), on_recog=go("recognition"), on_sos=go("sos"), fonts=fonts, sign_engine=self.sign_engine)
        self.pages["search"] = self.search_page
        # <<< 추가: search_page가 main_app의 정보(current_location)에 접근할 수 있도록 self 전달
        if hasattr(self.search_page, 'set_main_app'):
            self.search_page.set_main_app(self)

        self.sos_page = SOSPage(ASSETS, on_home=go("welcome"), on_nav=go("navigation"), on_send=go("recognition"))
        self.pages["sos"] = self.sos_page
        
        self.voice_page = VoicePage(ASSETS, fonts=fonts, sign_engine=self.sign_engine)
        self.pages["voice"] = self.voice_page
        
        for p in self.pages.values(): self.stack.addWidget(p)
        self.stack.setCurrentWidget(self.pages["welcome"])
        
        # --- 4. 중앙 시그널 연결 ---
        self.sign_engine.gesture_recognized.connect(self._handle_gesture)
        self.sign_engine.hangul_input_finished.connect(self._on_hangul_finished)
        self.sign_engine.session_finished.connect(self._on_session_finished)
        
        # GPS 신호를 중앙 관리 슬롯에 연결
        self.gps_client.new_location.connect(self._update_location)
        
        self.stack.currentChanged.connect(self._on_page_changed)
        
        # 수어->텍스트 변환 결과를 각 페이지의 채팅창에 연결
        try:
            self.sign_engine.hangul_input_finished.connect(self.pages["recognition"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["description"].append_user_text)
            self.sign_engine.hangul_input_finished.connect(self.pages["search"].append_user_text)
            self.sign_engine.hangul_result_updated.connect(self.voice_page._on_hangul_progress)
            self.sign_engine.hangul_input_finished.connect(self.voice_page.on_hangul_final)
            self.voice_page.playback_finished.connect(self._on_voice_playback_finished) # <<< 복원
        except Exception as e:
            print(f"[ERROR] Signal connection failed: {e}")

        # --- 5. 백그라운드 스레드 시작 ---
        self.engine_thread.start()
        self.gps_thread.start()
        print("✅ Sign Engine and GPS Client Threads started.")
        self._nav_pending = False

    # --- 중앙 제어 슬롯 메서드 ---
    @Slot(int)
    def _on_page_changed(self, index):
        current_page = self.stack.widget(index)
        page_name = type(current_page).__name__
        print(f"[App] Page changed to: {page_name}")

        if page_name in ("WelcomePage", "SOSPage"):
            self.sign_engine.switch_to_gesture_mode()
        elif page_name == "RecognitionPage":
            self._nav_pending = False 
            self.sign_engine.switch_to_gesture_mode()
            self.sign_engine.start_gesture_with_delay()
        else: # NavigationPage, DescriptionPage, SearchPage, VoicePage
            self.sign_engine.start_hangul_with_delay()

    # <<< 추가: VoicePage 음성 출력이 끝나면 제스처 모드로 전환
    @Slot()
    def _on_voice_playback_finished(self):
        print("[App] Voice playback finished. Switching to gesture mode.")
        if self.sign_engine:
            self.sign_engine.switch_to_gesture_mode()

    def _go_to_navigation(self): self.stack.setCurrentWidget(self.navigation_page)
    def _go_to_description(self): self.stack.setCurrentWidget(self.description_page)
    def _go_to_search(self): self.stack.setCurrentWidget(self.search_page)
    def _go_to_voice(self): self.stack.setCurrentWidget(self.voice_page)
    def _go_to_sos(self): self.stack.setCurrentWidget(self.sos_page)

    @Slot(str)
    def _handle_gesture(self, gesture: str):
        current_page = self.stack.currentWidget()
        page_name = type(current_page).__name__

        # <<< 추가: delete 제스처는 항상 인식 페이지로 돌아감
        if gesture == 'delete':
            print(f"[App] 'delete' gesture recognized! Returning to Recognition Page.")
            self.stack.setCurrentWidget(self.pages["recognition"])
            return

        if page_name == "WelcomePage" and gesture == ACTIVATION_GESTURE:
            print(f"[App] Activation gesture '{gesture}' detected! Switching to Recognition Page.")
            self.stack.setCurrentWidget(self.pages["recognition"])
            
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
            # <<< 수정: search_page는 main_app의 current_location을 직접 참조하므로 좌표 전달 불필요
            self.search_page.search_for(final_text)

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
            self.stack.setCurrentWidget(self.pages["welcome"])
        
    @Slot(float, float)
    def _update_location(self, lat, lng):
        """ <<< 수정: GPSClient로부터 새 위치를 받아 모든 관련 페이지에 전파하는 중앙 슬롯 """
        self.current_location = (lat, lng)
        
        # 각 페이지에 set_location 메서드가 있는지 확인하고 안전하게 호출
        if hasattr(self, 'sos_page') and hasattr(self.sos_page, 'set_location'):
            self.sos_page.set_location(lat, lng)
        if hasattr(self, 'navigation_page') and hasattr(self.navigation_page, 'set_location'):
            self.navigation_page.set_location(lat, lng)
        if hasattr(self, 'search_page') and hasattr(self.search_page, 'set_location'):
            self.search_page.set_location(lat, lng)

    def closeEvent(self, event):
        print("Main window closing. Shutting down all threads.")
        if hasattr(self, 'gps_client'): self.gps_client.stop()
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