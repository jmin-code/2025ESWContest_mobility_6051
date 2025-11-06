# ui/description.py
# -*- coding: utf-8 -*-
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
        self.on_home  = on_home
        self.on_recog = on_recog
        self.on_nav   = on_nav
        self.on_sos   = on_sos
        self.on_voice = on_voice
        self.fonts = fonts or {}
        self.sign_engine = sign_engine
        self.view_initialized = False

        # 배경
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "nav_bg.png")

        # 좌측 웹뷰
        self.web_view = QWebEngineView(self)
        self.web_view.setPage(QWebEnginePage(self))
        self.web_view.loadFinished.connect(self._start_hangul_input_timer)

        # 우측 패널
        self.camera_view = QLabel(self); self.camera_view.setStyleSheet("background-color: black;")
        self.lbl_hangul = QLabel("", self)
        self.lbl_hangul.setFont(QFont(self.fonts.get("korean", "Arial"), 22, QFont.Bold))
        self.lbl_hangul.setAlignment(Qt.AlignCenter)
        self.lbl_hangul.setWordWrap(True)
        self.lbl_hangul.setStyleSheet("color:#fff; background:rgba(0,0,0,.55); border-radius:10px; padding:6px 10px;")

        user_png = self.assets / "icons" / "user_bubble.png"
        bot_png  = self.assets / "icons" / "com_bubble.png"
        self.chat = ChatPanel(str(user_png), str(bot_png), parent=self)
        self.chat.setObjectName("descriptionChat")
        self.chat.hide()

        # 상단바
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            if cb: b.clicked.connect(cb)
            return b
        self.btn_home  = mk_icon("home.png",       self.on_home)
        self.btn_voice = mk_icon("voice_over.png", self.on_voice)  # 필요시 voice_over_b.png
        self.btn_nav   = mk_icon("nav_b.png",      self.on_nav)    # 파일명 확인
        self.btn_sos   = mk_icon("sos.png",        self.on_sos)

        # 엔진 시그널
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self.lbl_hangul.setText)

        # 800x480 기준(네비와 동일)
        self.layout = {
            "chat":   (520,  79, 260, 140),
            "input":  (520, 242, 265,  56),
            "camera": (520, 303, 260, 160),
        }
        self._relayout()

    # ===== 라이프사이클 =====
    def showEvent(self, event):
        super().showEvent(event)
        if not self.view_initialized:
            self.load_initial_view()
            self.view_initialized = True
        if self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    @Slot(bool)
    def _start_hangul_input_timer(self, ok):
        if ok and self.sign_engine:
            QTimer.singleShot(1000, self.sign_engine.switch_to_hangul_mode)

    # ===== 동작 =====
    def load_initial_view(self):
        self.web_view.load(QUrl("http://localhost:5050/info.html"))
        self.lbl_hangul.setText("무엇을 검색할까요?")

    def search_for(self, keyword: str):
        kw = (keyword or "").strip()
        if not kw: return
        self.web_view.load(QUrl(f"http://localhost:5050/info.html?keyword={kw}"))
        self.lbl_hangul.setText(f"검색: {kw}")
        try:
            self.chat.append(kw, role="user")
            self.chat.append("장소 정보를 검색합니다.", role="bot")
        except Exception:
            pass

    # ===== 카메라 =====
    @Slot(QImage)
    def set_camera_image(self, qt_image: QImage):
        if self.camera_view.isVisible() and self.camera_view.width() > 0:
            pm = QPixmap.fromImage(qt_image)
            self.camera_view.setPixmap(pm.scaled(self.camera_view.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))

    # ===== 유틸 =====
    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W*2 or pm.height() >= self.BASE_H*2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0, h0 = int(pm.width()/dpr), int(pm.height()/dpr)
        bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)): return box
        s = min(bw/w0, bh/h0); w, h = int(w0*s), int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    # ★ 네비게이션과 동일한 서명/동작
    def _map_from_design(self, fit: QRect, x, y, w=None, h=None, *, right=None, bottom=None) -> QRect:
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W = fit.width()-X+fit.x()-int(round((right or 0)*sx))
            H = fit.height()-Y+fit.y()-int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)

    def _rect(self, fit: QRect, key: str) -> QRect:
        x, y, w, h = self.layout[key]
        return self._map_from_design(fit, x, y, w=w, h=h)

    # ===== 레이아웃 =====
    def _relayout(self):
        full = self.rect()
        self.bg.setGeometry(full)

        # 배경을 fit 박스로 스케일(네비와 동일)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width()*dpr), int(fit.height()*dpr),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img); pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)

        # 좌측 웹뷰 — 네비와 동일 좌표 (22, 87, 468, 373)
        self.web_view.setGeometry(self._map_from_design(fit, 22, 87, w=468, h=373))

        # 우측 패널
        self.chat.setGeometry(self._rect(fit, "chat"))
        if not self.chat.isVisible(): self.chat.show()
        self.lbl_hangul.setGeometry(self._rect(fit, "input"))
        self.camera_view.setGeometry(self._rect(fit, "camera"))

        # 상단 아이콘 (fit 기준으로 매핑)
        for btn, x, y0, w0, h0 in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(fit, x, y0, w=w0, h=h0)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

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
