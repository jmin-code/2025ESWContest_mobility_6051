# ui/navigation.py
from PySide6.QtCore import Qt, QRect, QSize, QUrl, Slot, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from ui.chat import ChatPanel  # ← 추가

class NavigationPage(QWidget):
    BASE_W, BASE_H = 800, 480

    def __init__(self, assets_dir, on_home=None, on_nav=None, on_sos=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir; self.on_home = on_home; self.on_nav = on_nav; self.on_sos = on_sos
        self.fonts = fonts or {}; self.sign_engine = sign_engine
        self.seoul_station_coords = (37.554678, 126.970609)
        self.map_initialized = False

        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        self.map_view = QWebEngineView(self)
        self.map_view.setPage(QWebEnginePage(self))
        self.map_view.loadFinished.connect(self._start_hangul_input_timer)

        self.camera_view = QLabel(self); self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        # self.lbl_hangul.setAlignment(Qt.AlignCenter)
        # self.lbl_hangul.setStyleSheet("color: white; background: rgba(0,0,0,0.6); border-radius: 8px;")
        self.lbl_hangul.setAlignment(Qt.AlignCenter)
        self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet(
            "color: white; background: rgba(0,0,0,0.55); border-radius: 10px; padding: 6px 10px;"
        )

        # --- ChatPanel (우측 상단 채팅 박스: 수어→텍스트 기록) ---
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

        self.btn_home = mk_icon("home.png", self.on_home)
        self.btn_nav  = mk_icon("nav_b.png", lambda: None)
        self.btn_sos  = mk_icon("sos.png", self.on_sos)

        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)  # 입력 중 미리보기


        self.layout = {
            "chat":   (520,  79, 260, 140),  # 우측 상단 채팅
            "input":  (520, 240, 265,  56),  # ← "목적지를 입력…" 실시간 텍스트 박스 (조절 포인트)
            "camera": (520, 303, 260, 160),  # ← 카메라 더 작고, 더 아래 (조절 포인트)
        }
        self._relayout()
        

    def showEvent(self, event):
        super().showEvent(event)
        if not self.map_initialized:
            print("[NavPage] Showing. Loading initial map.")
            self.load_initial_map()
            self.map_initialized = True

        print("[NavPage] Page is visible. Starting 1-second timer for Hangul input.")
        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(bool)
    def _start_hangul_input_timer(self, ok):
        if ok:
            print("[NavPage] Map loading finished. Starting 1-second timer for Hangul input.")
            if self.sign_engine:
                QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)
        else:
            print("[NavPage] Map loading failed.")

    def load_initial_map(self):
        lat, lng = self.seoul_station_coords
        base = "http://localhost:5050/map.html"
        url = QUrl(f"{base}?sLat={lat}&sLng={lng}")
        self.map_view.load(url)
        self.lbl_hangul.setText("목적지를 입력하세요")

    def update_route(self, destination: str):
        if not destination: return
        print(f"[NavPage] Updating route to: {destination}")
        js_code = f"drawRouteToDestination('{destination}');"
        self.map_view.page().runJavaScript(js_code)
        self.lbl_hangul.setText(f"경로: {destination}")

        # 채팅 기록(사용자 입력 + 시스템 응답)
        try:
            self.chat.append(destination.strip(), role="user")
            self.chat.append("경로 안내를 시작합니다.", role="bot")
        except Exception:
            pass

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

    def _map_from_design(self, fit, x, y, w=None, h=None, *, right=None, bottom=None):
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W = fit.width()-X+fit.x()-int(round((right or 0)*sx))
            H = fit.height()-Y+fit.y()-int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)
    
    # 레이아웃 헬퍼: 키로 가져오기
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

        # 좌: 지도
        self.map_view.setGeometry(self._map_from_design(fit, 15, 79, w=480, h=385))

        # 우상: 채팅(이미지와 동일한 스타일 영역 상단)
        # chat_rect = self._map_from_design(fit, 520, 79, w=260, h=140)  # 디자인 기준: 우측 상단 블록
        # self.chat.setGeometry(chat_rect)
        # 우상: 채팅
        self.chat.setGeometry(self._rect(fit, "chat"))
        if not self.chat.isVisible(): self.chat.show()
        self.chat.raise_()

        # 우중: 진행 중 텍스트(보조 라벨) — 채팅 아래
        self.lbl_hangul.setGeometry(self._map_from_design(fit, 520, 190, w=260, h=40))
        # 우중: 입력 라벨(“목적지를 입력하세요” + 실시간 텍스트)
        self.lbl_hangul.setGeometry(self._rect(fit, "input"))
        # 필요 시 표시 유지/숨김 전환 가능
        # self.lbl_hangul.setVisible(False)  # 채팅만 쓰고 싶으면 주석 해제

        # 우하: 카메라 프리뷰
        # self.camera_view.setGeometry(self._map_from_design(fit, 526, 245, w=254, h=200))
        self.camera_view.setGeometry(self._rect(fit, "camera"))

        # 아이콘 버튼
        for btn, x, y, w, h in (
            (self.btn_home, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(fit, x, y, w=w, h=h)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

        # z-order 정리
        self.bg.lower()
        self.map_view.raise_()
        self.chat.raise_()
        self.lbl_hangul.raise_()
        self.camera_view.raise_()
        for b in (self.btn_home, self.btn_nav, self.btn_sos):
            b.raise_()

    def resizeEvent(self, e):
        self._relayout(); super().resizeEvent(e)