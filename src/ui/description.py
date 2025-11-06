from PySide6.QtCore import Qt, QRect, QSize, QUrl, Slot, QTimer
from PySide6.QtGui import QPixmap, QIcon, QImage, QFont
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage
from ui.chat import ChatPanel

class DescriptionPage(QWidget):
    BASE_W, BASE_H = 800, 480

    def __init__(self, assets_dir, on_home=None, on_recog=None, on_nav=None, on_sos=None, on_voice=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir
        self.on_home = on_home
        self.on_recog = on_recog
        self.on_sos = on_sos
        self.on_nav = on_nav
        self.on_voice = on_voice
        self.fonts = fonts or {}
        self.sign_engine = sign_engine
        self.view_initialized = False

        # 배경
        self.bg = QLabel(self)
        self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        # 웹뷰
        self.web_view = QWebEngineView(self)
        self.web_view.setPage(QWebEnginePage(self))
        self.web_view.loadFinished.connect(self._start_hangul_input_timer)

        # 카메라/한글 진행 표시
        self.camera_view = QLabel(self)
        self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        self.lbl_hangul.setAlignment(Qt.AlignCenter)
        self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet("color: white; background: rgba(0,0,0,0.55); border-radius: 10px; padding: 6px 10px;")

        # 채팅 패널
        user_png = self.assets / "icons" / "user_bubble.png"
        bot_png  = self.assets / "icons" / "com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("descriptionChat")
        self.chat.hide()

        # 상단 아이콘
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull():
                b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            if cb:
                b.clicked.connect(cb)
            return b

        self.btn_home  = mk_icon("home.png", self.on_home)
        self.btn_voice = mk_icon("voice.png", self.on_voice)           # 필요 시 voice_over_b.png 로 교체
        self.btn_nav   = mk_icon("nav_b.png", self.on_nav)             # 아이콘 파일명 확인 필요
        self.btn_sos   = mk_icon("sos.png", self.on_sos)

        # 제스처 엔진 연결
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)

        # 레이아웃 좌표(디자인 기준 800x480)
        self.layout = {
            "chat":   (520,  79, 260, 140),
            "input":  (520, 240, 265,  56),
            "camera": (520, 303, 260, 160),
        }

        self._relayout()

    def showEvent(self, event):
        super().showEvent(event)
        if not self.view_initialized:
            print("[DescPage] Showing. Loading initial view.")
            self.load_initial_view()
            self.view_initialized = True

        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(bool)
    def _start_hangul_input_timer(self, ok):
        if ok:
            print("[DescPage] View loading finished. Starting 1-second timer for Hangul input.")
            if self.sign_engine:
                QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)
        else:
            print("[DescPage] View loading failed.")

    def load_initial_view(self):
        base = "http://localhost:5050/info.html"
        url = QUrl(base)
        self.web_view.load(url)
        self.lbl_hangul.setText("무엇을 검색할까요?")

    def search_for(self, keyword: str):
        if not keyword:
            return
        print(f"[DescPage] Searching for: {keyword}")
        base = "http://localhost:5050/info.html"
        url = QUrl(f"{base}?keyword={keyword}")
        self.web_view.load(url)
        self.lbl_hangul.setText(f"검색: {keyword}")

        try:
            self.chat.append(keyword.strip(), role="user")
            self.chat.append("장소 정보를 검색합니다.", role="bot")
        except Exception:
            pass

    @Slot(QImage)
    def set_camera_image(self, qt_image: QImage):
        if self.camera_view.isVisible() and self.camera_view.width() > 0:
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.camera_view.setPixmap(scaled_pixmap)

    # ====== 유틸 ======
    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W * 2 or pm.height() >= self.BASE_H * 2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull():
            return box
        dpr = pm.devicePixelRatio() or 1.0
        w0, h0 = int(pm.width() / dpr), int(pm.height() / dpr)
        bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)):
            return box
        s = min(bw / w0, bh / h0)
        w, h = int(w0 * s), int(h0 * s)
        return QRect(box.x() + (bw - w) // 2, box.y() + (bh - h) // 2, w, h)

    def _map_from_design(self, x, y, w, h, *, out_rect: QRect) -> QRect:
        """800x480 기준 좌표(x,y,w,h)를 실제 창 out_rect로 매핑"""
        sx = out_rect.width()  / self.BASE_W
        sy = out_rect.height() / self.BASE_H
        X = int(round(x * sx))
        Y = int(round(y * sy))
        W = int(round(w * sx))
        H = int(round(h * sy))
        return QRect(X, Y, W, H)

    def _rect(self, out_rect: QRect, key: str) -> QRect:
        x, y, w, h = self.layout[key]
        return self._map_from_design(x, y, w, h, out_rect=out_rect)

    # ====== 레이아웃 ======
    def _relayout(self):
        full = self.rect()

        # 배경
        self.bg.setGeometry(full)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width() * dpr), int(fit.height() * dpr),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img)
            pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)

        # 웹뷰: (15, 79, 480, 385) -> 전체 영역 기준으로 매핑
        self.web_view.setGeometry(self._map_from_design(15, 79, 480, 385, out_rect=full))

        # 우측 패널들
        self.chat.setGeometry(self._rect(full, "chat"))
        if not self.chat.isVisible():
            self.chat.show()

        self.lbl_hangul.setGeometry(self._rect(full, "input"))
        self.camera_view.setGeometry(self._rect(full, "camera"))

        # 상단 아이콘들
        for btn, x, y0, w0, h0 in (
            (self.btn_home,  603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,   703, 20, 22, 22),
            (self.btn_sos,   753, 20, 22, 22),
        ):
            rbtn = self._map_from_design(x, y0, w0, h0, out_rect=full)
            btn.setFixedSize(rbtn.width(), rbtn.height())
            btn.setIconSize(QSize(rbtn.width(), rbtn.height()))
            btn.move(rbtn.x(), rbtn.y())

        # Z-Order
        self.bg.lower()
        self.web_view.raise_()
        self.chat.raise_()
        self.lbl_hangul.raise_()
        self.camera_view.raise_()
        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
            b.raise_()

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)
