# ui/search.py (수정된 전체 코드)

from PySide6.QtCore import Qt, QRect, QSize, QUrl, Slot, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor
)
from ui.chat import ChatPanel
import time, urllib.parse

# --- 디버깅용 클래스 (기존과 동일) ---
class _NetLogger(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        url = info.requestUrl().toString()
        if any(k in url for k in ("dapi.kakao.com", "map.daumcdn.net")):
            print(f"[WEB REQ] {url}")

class _DebugWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        lvl = {0:"Info", 1:"Warning", 2:"Error"}.get(level, str(level))
        print(f"[WEB CONSOLE|{lvl}] {sourceID}:{lineNumber} — {message}")
# ------------------------------------

class SearchPage(QWidget):
    BASE_W, BASE_H = 800, 480

    def __init__(self, assets_dir, on_home=None, on_recog=None, on_sos=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir
        self.on_home = on_home
        self.on_recog = on_recog
        self.on_sos = on_sos
        self.fonts = fonts or {}
        self.sign_engine = sign_engine
        self.view_initialized = False
        
        self.is_map_locked = False
        self.current_location = None
        self._map_ready = False
        # <<< 추가: 실시간 GPS 좌표를 저장할 변수
        self.current_location = None 
        self._map_ready = False

        # --- 배경 ---
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        # --- WebEngine 설정 (기존과 동일) ---
        self.web_view = QWebEngineView(self)
        profile = QWebEngineProfile.defaultProfile()
        profile.setUrlRequestInterceptor(_NetLogger(self))
        page = _DebugWebPage(profile, self)
        page.featurePermissionRequested.connect(
            lambda origin, feature: page.setFeaturePermission(origin, feature, QWebEnginePage.PermissionGrantedByUser)
        )
        self.web_view.setPage(page)
        self.web_view.loadFinished.connect(self._on_map_loaded) # loadFinished 시그널 변경

        # --- 카메라/라벨/채팅 (기존과 동일) ---
        self.camera_view = QLabel(self); self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        self.lbl_hangul.setAlignment(Qt.AlignCenter); self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet("color:#fff;background:rgba(0,0,0,.55);border-radius:10px;padding:6px 10px;")
        user_png = self.assets / "icons" / "user_bubble.png"
        bot_png  = self.assets / "icons" / "com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("searchChat"); self.chat.hide()
        
        # --- 버튼 (기존과 동일) ---
        def mk_icon(fname, cb):
            b = QPushButton(self); pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent"); b.setCursor(Qt.PointingHandCursor)
            if cb: b.clicked.connect(cb)
            return b
        self.btn_home  = mk_icon("home.png", self.on_home)
        self.btn_recog = mk_icon("nav_b.png", self.on_recog) # nav_b.png가 맞는 아이콘인지 확인 필요
        self.btn_sos   = mk_icon("sos.png", self.on_sos)

        # --- 시그널 연결 (기존과 동일) ---
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)

        self.btn_recenter = QPushButton("현위치", self)
        self.btn_recenter.setStyleSheet("""
            QPushButton { 
                background-color: rgba(255, 255, 255, 0.9); color: #333; 
                border: 1px solid #aaa; border-radius: 8px; 
                padding: 8px 12px; font-size: 14px;
            }
            QPushButton:pressed { background-color: #e0e0e0; }
        """)
        self.btn_recenter.setCursor(Qt.PointingHandCursor)
        self.btn_recenter.clicked.connect(self._unlock_and_recenter_map)
        self.btn_recenter.hide()
        self.layout = { "chat":(520,79,260,140), "input":(520,240,265,56), "camera":(520,303,260,160) }
        self._relayout()

    def showEvent(self, e):
        super().showEvent(e)
        # <<< 수정: 페이지가 보일 때마다 지도 고정 해제 및 초기화
        self.is_map_locked = False
        self.btn_recenter.hide()
        
        if not self.view_initialized:
            print("[SearchPage] Showing → load_initial_view()")
            self.load_initial_view()
            self.view_initialized = True
        
        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(bool)
    def _on_map_loaded(self, ok):
        if not ok: return
        self._map_ready = True
        print("[SearchPage] 맵 로드 완료. 초기 위치 주입 시도.")
        if self.current_location:
            self.set_location(*self.current_location)
            
    def load_initial_view(self):
        self.web_view.load(QUrl("http://localhost:5050/search.html"))
        self.lbl_hangul.setText("주변 인프라 탐색")

    # <<< 추가: main.py로부터 실시간 GPS 좌표를 받는 슬롯
    @Slot(float, float)
    def set_location(self, lat: float, lng: float):
        # <<< 수정: 항상 현재 위치는 저장하되, 지도 고정 시 UI 업데이트는 생략
        self.current_location = (lat, lng)
        if self.is_map_locked:
            return

        if self._map_ready:
            js_code = f"window.updateCurrentLocation({lat}, {lng});"
            self.web_view.page().runJavaScript(js_code)

    def search_for(self, keyword: str):
        if not self._map_ready or not self.current_location:
            # 준비 안됐으면 검색 불가
            return

        # <<< 추가: 검색 시작 시 지도 고정 및 현위치 버튼 표시
        self.is_map_locked = True
        self.btn_recenter.show()
            
        lat, lng = self.current_location
        keyword = keyword.strip()
        print(f"[SearchPage] SEARCH keyword='{keyword}' at ({lat}, {lng}) and Lock Map")
        
        js_code = f"runNearbySearch('{keyword}', {lat}, {lng});"
        self.web_view.page().runJavaScript(js_code)
        self.lbl_hangul.setText(f"검색: {keyword}")
        try:
            self.chat.append(keyword, role="user")
            self.chat.append("주변을 탐색합니다.", role="bot")
        except Exception:
            pass

    @Slot(QImage)
    def set_camera_image(self, qt_image: QImage):
        if self.camera_view.isVisible() and self.camera_view.width() > 0:
            pm = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pm.scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    # === 유틸/레이아웃 (이하 기존과 동일) ===
    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W*2 or pm.height() >= self.BASE_H*2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0,h0=int(pm.width()/dpr),int(pm.height()/dpr);bw,bh=box.width(),box.height()
        if not all((w0,h0,bw,bh)): return box
        s=min(bw/w0,bh/h0);w,h=int(w0*s),int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    def _map_from_design(self, fit, x, y, w=None, h=None, *, right=None, bottom=None):
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W=fit.width()-X+fit.x()-int(round((right or 0)*sx));H=fit.height()-Y+fit.y()-int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)

    def _rect(self, fit, key):
        x, y, w, h = self.layout[key]
        return self._map_from_design(fit, x, y, w=w, h=h)

    @Slot()
    def _unlock_and_recenter_map(self):
        self.is_map_locked = False
        self.btn_recenter.hide()
        self.lbl_hangul.setText("주변 인프라 탐색")
        
        if self.current_location:
            print("[SearchPage] 지도 고정 해제 및 현위치로 복귀")
            # 저장된 최신 위치로 즉시 이동하고, 키워드 검색은 초기화
            js_code = f"runNearbySearch('', {self.current_location[0]}, {self.current_location[1]});"
            self.web_view.page().runJavaScript(js_code)
            self.set_location(*self.current_location)

    def _relayout(self):
        full=self.rect();self.bg.setGeometry(full);fit=self._fit_rect_for_pixmap(self.pm_bg,full)
        if not self.pm_bg.isNull():
            dpr=self.pm_bg.devicePixelRatio()or 1.0;img=self.pm_bg.toImage().scaled(int(fit.width()*dpr),int(fit.height()*dpr),Qt.KeepAspectRatio,Qt.SmoothTransformation)
            pm2=QPixmap.fromImage(img);pm2.setDevicePixelRatio(dpr);self.bg.setPixmap(pm2)
        self.web_view.setGeometry(self._map_from_design(fit,15,79,w=480,h=385));self.chat.setGeometry(self._rect(fit,"chat"))
        if not self.chat.isVisible():self.chat.show()
        self.lbl_hangul.setGeometry(self._rect(fit,"input"));self.camera_view.setGeometry(self._rect(fit,"camera"))
        for btn,x,y,w,h in((self.btn_home,653,20,24,24),(self.btn_recog,703,20,22,22),(self.btn_sos,753,20,22,22)):
            r=self._map_from_design(fit,x,y,w=w,h=h)
            btn.setIconSize(QSize(r.width(),r.height()));btn.setFixedSize(r.width(),r.height());btn.move(r.x(),r.y())
        self.bg.lower();self.web_view.raise_();self.chat.raise_();self.lbl_hangul.raise_();self.camera_view.raise_()
        self.btn_recenter.move(30, 95)
        self.btn_recenter.raise_()
        for b in(self.btn_home,self.btn_recog,self.btn_sos):b.raise_()

    def resizeEvent(self, e):
        self._relayout();super().resizeEvent(e)