# ui/search.py
# -*- coding: utf-8 -*-

from PySide6.QtCore import Qt, QRect, QSize, QUrl, Slot, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import (
    QWebEnginePage, QWebEngineProfile, QWebEngineSettings, QWebEngineUrlRequestInterceptor
)
from ui.chat import ChatPanel
import urllib.parse

# --- 디버깅용 클래스 ---
class _NetLogger(QWebEngineUrlRequestInterceptor):
    def interceptRequest(self, info):
        url = info.requestUrl().toString()
        if any(k in url for k in ("dapi.kakao.com", "map.daumcdn.net")):
            print(f"[WEB REQ] {url}")

class _DebugWebPage(QWebEnginePage):
    def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):
        lvl = {0:"Info", 1:"Warning", 2:"Error"}.get(level, str(level))
        print(f"[WEB CONSOLE|{lvl}] {sourceID}:{lineNumber} — {message}")

class SearchPage(QWidget):
    BASE_W, BASE_H = 800, 480

    def __init__(self, assets_dir, on_home=None, on_voice=None, on_recog=None, on_nav=None, on_sos=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir
        self.on_home  = on_home
        self.on_voice = on_voice
        self.on_recog = on_recog
        self.on_nav   = on_nav
        self.on_sos   = on_sos
        self.fonts = fonts or {}
        self.sign_engine = sign_engine

        self.view_initialized = False
        self.is_map_locked = False
        self.current_location = (37.576738, 126.897859)
        self._map_ready = False

        # --- 배경 ---
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        # --- WebEngine 설정 ---
        self.web_view = QWebEngineView(self)
        profile = QWebEngineProfile.defaultProfile()
        self._interceptor = _NetLogger(self)  # GC 방지 위해 보관
        profile.setUrlRequestInterceptor(self._interceptor)

        page = _DebugWebPage(profile, self)
        page.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        page.featurePermissionRequested.connect(
            lambda origin, feature: page.setFeaturePermission(origin, feature, QWebEnginePage.PermissionGrantedByUser)
        )
        self.web_view.setPage(page)
        self.web_view.loadFinished.connect(self._on_map_loaded)

        # --- 카메라/라벨/채팅 ---
        self.camera_view = QLabel(self); self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        self.lbl_hangul.setAlignment(Qt.AlignCenter)
        self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet("color:#fff;background:rgba(0,0,0,.55);border-radius:10px;padding:6px 10px;")

        user_png = self.assets / "icons" / "user_bubble.png"
        bot_png  = self.assets / "icons" / "com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("searchChat")
        self.chat.hide()

        # --- 상단바 버튼 ---
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull():
                b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            if cb: b.clicked.connect(cb)
            return b
        self.btn_home  = mk_icon("home.png",        self.on_home)
        self.btn_voice = mk_icon("voice_over.png",  self.on_voice)
        self.btn_nav   = mk_icon("nav_b.png",       self.on_nav)   # 활성 탭
        self.btn_sos   = mk_icon("sos.png",         self.on_sos)

        # --- 엔진 시그널 ---
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)

        # --- 현위치 버튼 ---
        self.btn_recenter = QPushButton("현위치", self)
        self.btn_recenter.setStyleSheet("""
            QPushButton {
                color: #111;
                background: #EAEAEA;
                border: 1px solid #CFCFCF;
                border-radius: 10px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:hover { background: #F3F3F3; }
            QPushButton:pressed { background: #E0E0E0; }
        """)
        self.btn_recenter.setCursor(Qt.PointingHandCursor)
        self.btn_recenter.clicked.connect(self._recenter_map_once)
        self.btn_recenter.hide()

        # --- 디자인 기준 좌표 (Navigation과 동일 방식) ---
        self.layout = {
            "chat":   (520,  79, 260, 140),
            "input":  (520, 242, 265,  56),
            "camera": (520, 303, 265, 160),
        }
        self._relayout()

    # ===== 라이프사이클 =====
    def showEvent(self, e):
        super().showEvent(e)
        # 페이지 진입 시 상태 초기화
        self.is_map_locked = False
        self.btn_recenter.hide()
        self.lbl_hangul.setText("주변 인프라 탐색")

        if not self.view_initialized:
            self.load_initial_view()
            self.view_initialized = True
        elif self.current_location:
            self.set_location(*self.current_location)

        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(bool)
    def _on_map_loaded(self, ok):
        if not ok:
            print("[SearchPage] 맵 로드 실패")
            return
        self._map_ready = True
        print("[SearchPage] 맵 로드 완료. 초기 위치 주입 시도.")
        if self.current_location:
            self.set_location(*self.current_location)

    def load_initial_view(self):
        self.web_view.load(QUrl("http://localhost:5050/search.html"))
        self.lbl_hangul.setText("주변 인프라 탐색")

    # === main.py 로부터 GPS 좌표 수신 ===
    @Slot(float, float)
    def set_location(self, lat: float, lng: float):
        self.current_location = (lat, lng)
        if self.is_map_locked:
            return  # 검색 고정 상태에서는 자동 이동 안 함
        if self._map_ready:
            js_code = f"window.updateCurrentLocation({lat}, {lng});"
            self.web_view.page().runJavaScript(js_code)

    # === 검색 ===
    def search_for(self, keyword: str):
        if not self._map_ready or not self.current_location:
            return
        self.is_map_locked = True
        self.btn_recenter.show()

        lat, lng = self.current_location
        safe_kw = (keyword or "").strip()
        # js_code = f"runNearbySearch({safe_kw!r}, {lat}, {lng});"
        js_code = f"runNearbySearch('{keyword.strip()}', {lat}, {lng});"
        self.web_view.page().runJavaScript(js_code)
        self.lbl_hangul.setText(f"검색: {keyword}")

        try:
            self.chat.append(safe_kw, role="user")
            self.chat.append("주변을 탐색합니다.", role="bot")
        except Exception:
            pass

    # === 카메라 피드 ===
    @Slot(QImage)
    def set_camera_image(self, qt_image: QImage):
        if self.camera_view.isVisible() and self.camera_view.width() > 0:
            pm = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pm.scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    # ===== 유틸 (NavigationPage와 동일한 형태) =====
    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W*2 or pm.height() >= self.BASE_H*2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0,h0 = int(pm.width()/dpr), int(pm.height()/dpr)
        bw,bh = box.width(), box.height()
        if not all((w0,h0,bw,bh)): return box
        s = min(bw/w0, bh/h0); w,h = int(w0*s), int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    def _map_from_design(self, fit: QRect, x, y, w=None, h=None, *, right=None, bottom=None) -> QRect:
        """Navigation과 동일: fit(배경이 들어간 사각형) 기준으로 800x480 좌표 매핑"""
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W = fit.width()  - (X - fit.x()) - int(round((right or 0)*sx))
            H = fit.height() - (Y - fit.y()) - int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)

    def _rect(self, fit: QRect, key: str) -> QRect:
        x, y, w, h = self.layout[key]
        return self._map_from_design(fit, x, y, w=w, h=h)

    # === 지도 컨트롤 ===
    @Slot()
    def _recenter_map_once(self):
        """지도 고정은 유지한 채, 현재 위치로 1회만 부드럽게 이동"""
        if self.current_location:
            lat, lng = self.current_location
            print("[SearchPage] 지도 고정 상태에서 현위치로 1회 이동")
            self.web_view.page().runJavaScript(f"panToLocation({lat}, {lng});")

    @Slot()
    def _unlock_and_recenter_map(self):
        """검색 고정을 해제하고, 현위치 중심으로 다시 추적 시작"""
        self.is_map_locked = False
        self.btn_recenter.hide()
        self.lbl_hangul.setText("주변 인프라 탐색")
        if self.current_location:
            lat, lng = self.current_location
            self.web_view.page().runJavaScript(f"runNearbySearch('', {lat}, {lng});")
            self.set_location(lat, lng)

    # ===== 레이아웃 =====
    def _relayout(self):
        full = self.rect()

        # 배경 fit 사각형
        self.bg.setGeometry(full)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width()*dpr), int(fit.height()*dpr),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img); pm2.setDevicePixelRatio(dpr)
            self.bg.setGeometry(fit)
            self.bg.setPixmap(pm2)

        # 좌측 지도 (Navigation과 동일 좌표)
        self.web_view.setGeometry(self._map_from_design(fit, 22, 87, w=468, h=373))

        # 우측 패널
        self.chat.setGeometry(self._rect(fit, "chat"))
        if not self.chat.isVisible(): self.chat.show()
        self.lbl_hangul.setGeometry(self._rect(fit, "input"))
        self.camera_view.setGeometry(self._rect(fit, "camera"))

        # 상단 아이콘 (Navigation 동일)
        for btn, x, y0, w0, h0 in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            rbtn = self._map_from_design(fit, x, y0, w=w0, h=h0)
            btn.setFixedSize(rbtn.width(), rbtn.height())
            btn.setIconSize(QSize(rbtn.width(), rbtn.height()))
            btn.move(rbtn.x(), rbtn.y())

        # 현위치 버튼 (Navigation과 동일 위치 고정)
        self.btn_recenter.move(30, 95)
        self.btn_recenter.raise_()

        # Z-Order
        self.bg.lower()
        self.web_view.raise_()
        self.chat.raise_()
        self.lbl_hangul.raise_()
        self.camera_view.raise_()
        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos, self.btn_recenter):
            b.raise_()

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)
