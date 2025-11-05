
from PySide6.QtCore import Qt, QRect, QSize, QUrl, Slot, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from ui.chat import ChatPanel

class NavigationPage(QWidget):
    BASE_W, BASE_H = 800, 480

    def __init__(self, assets_dir, on_home=None, on_voice=None, on_nav=None, on_sos=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir; self.on_home = on_home; self.on_voice = on_voice; self.on_nav = on_nav; self.on_sos = on_sos
        self.fonts = fonts or {}; self.sign_engine = sign_engine
        self.seoul_station_coords = (37.576738, 126.897859 )
        self.map_initialized = False
        self.current_location = None

        self.is_map_locked = False

        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        self.map_view = QWebEngineView(self)
        self.map_view.setPage(QWebEnginePage(self))

        self.camera_view = QLabel(self); self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        self.lbl_hangul.setAlignment(Qt.AlignCenter)
        self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet(
            "color: white; background: rgba(0,0,0,0.55); border-radius: 10px; padding: 6px 10px;"
        )

        user_png = self.assets / "icons" / "user_bubble.png"
        bot_png  = self.assets / "icons" / "com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("navigationChat")
        self.chat.hide()

        def mk_icon(fname, cb):
            b = QPushButton(self); pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent"); b.setCursor(Qt.PointingHandCursor); b.clicked.connect(cb)
            return b

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
        self.btn_recenter.clicked.connect(self._recenter_map_once) 
        self.btn_recenter.hide()

        self.btn_home = mk_icon("home.png", self.on_home)
        self.btn_voice = mk_icon("voice_over.png", self.on_voice)
        self.btn_nav  = mk_icon("nav_b.png", lambda: None)
        self.btn_sos  = mk_icon("sos.png", self.on_sos)

        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)

        self.layout = {
            "chat":   (520,  79, 260, 140),  # 우측 상단 채팅
            "input":  (520, 242, 265,  56),  # ← "목적지를 입력…" 실시간 텍스트 박스 (조절 포인트)
            "camera": (520, 303, 265, 160),  # ← 카메라 더 작고, 더 아래 (조절 포인트)
        }
        self._relayout()

    def showEvent(self, event):
        super().showEvent(event)
        # 페이지가 보일 때마다 지도 고정 해제 및 실시간 추적 모드로 초기화
        self.is_map_locked = False 
        self.btn_recenter.hide()
        self.lbl_hangul.setText("목적지를 입력하세요")
        self.load_initial_map()
        
        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(float, float)
    def set_location(self, lat: float, lng: float):
        self.current_location = (lat, lng)
        if self.is_map_locked:
            return # 고정 상태에서는 지도 자동 이동 안함

        if self.map_initialized:
            js_code = f"setStartLocation({lat}, {lng});"
            self.map_view.page().runJavaScript(js_code)
            
    def load_initial_map(self):
        start_lat, start_lng = self.current_location if self.current_location else self.seoul_station_coords
        base = "http://localhost:5050/map.html"
        url = QUrl(f"{base}?sLat={start_lat}&sLng={start_lng}")
        self.map_view.load(url)
        self.map_initialized = True


    def update_route(self, destination: str):
        """입력된 목적지로 경로 탐색을 요청합니다."""
        if not destination: return

        self.is_map_locked = True
        self.btn_recenter.show()
        
        js_code = f"drawRouteToDestination('{destination}');"
        self.map_view.page().runJavaScript(js_code)
        self.lbl_hangul.setText(f"경로: {destination}")
        
        print(f"[NavPage] '{destination}' 경로 탐색 요청")
        
        # map.html의 drawRouteToDestination 함수를 호출
        js_code = f"drawRouteToDestination('{destination}');"
        self.map_view.page().runJavaScript(js_code)
        self.lbl_hangul.setText(f"경로: {destination}")

        try:
            self.chat.append(destination.strip(), role="user")
            self.chat.append("경로 안내를 시작합니다.", role="bot")
        except Exception:
            pass

    # ... (set_camera_image, _load_pix, 레이아웃 관련 메서드는 기존과 동일)
    @Slot(QImage)
    def set_camera_image(self, qt_image: QImage):
        if self.camera_view.isVisible() and self.camera_view.width() > 0:
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)

    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W * 2 or pm.height() >= self.BASE_H * 2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0, h0 = int(pm.width()/dpr), int(pm.height()/dpr); bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)): return box
        s = min(bw/w0, bh/h0); w, h = int(w0*s), int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    @Slot()
    def _unlock_and_recenter_map(self):
        self.is_map_locked = False
        self.btn_recenter.hide()
        self.lbl_hangul.setText("목적지를 입력하세요")
        
        if self.current_location:
            print("[NavPage] 지도 고정 해제 및 현위치로 복귀")
            self.set_location(self.current_location[0], self.current_location[1])
        else:
            self.load_initial_map()

    @Slot()
    def _recenter_map_once(self):
        """지도 고정 상태는 유지한 채, 현재 위치로 지도를 한번만 이동시킴"""
        if self.current_location:
            print("[NavPage] 지도 고정 상태에서 현위치로 1회 이동")
            js_code = f"panToLocation({self.current_location[0]}, {self.current_location[1]});"
            self.map_view.page().runJavaScript(js_code)

            
    def _map_from_design(self, fit, x, y, w=None, h=None, *, right=None, bottom=None):
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W = fit.width()-X+fit.x()-int(round((right or 0)*sx))
            H = fit.height()-Y+fit.y()-int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)
    
    def _rect(self, fit, key: str) -> QRect:
        x, y, w, h = self.layout[key]
        return self._map_from_design(fit, x, y, w=w, h=h)

    def _relayout(self):
        full = self.rect(); self.bg.setGeometry(full)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width()*dpr), int(fit.height()*dpr),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img); pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)

        self.map_view.setGeometry(self._map_from_design(fit, 22, 87, w=468, h=373))
        self.chat.setGeometry(self._rect(fit, "chat"))
        if not self.chat.isVisible(): self.chat.show()
        self.chat.raise_()
        self.lbl_hangul.setGeometry(self._map_from_design(fit, 520, 190, w=260, h=40))
        self.lbl_hangul.setGeometry(self._rect(fit, "input"))
        self.camera_view.setGeometry(self._rect(fit, "camera"))

        # 아이콘 버튼
        for btn, x, y, w, h in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(fit, x, y, w=w, h=h)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

        self.bg.lower()
        self.map_view.raise_()
        self.chat.raise_()
        self.lbl_hangul.raise_()
        self.camera_view.raise_()
        self.btn_recenter.move(30, 95)
        self.btn_recenter.raise_()
        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
            b.raise_()

    def resizeEvent(self, e):
        self._relayout(); super().resizeEvent(e)