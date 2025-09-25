from PySide6.QtCore import Qt, QSize, QRect, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtWidgets import QWidget, QLabel, QPushButton

class WelcomePage(QWidget):
    BASE_W, BASE_H = 800, 480
    BTN_W, BTN_H = 162, 49
    BTN_BOTTOM_MARGIN = 125

    def __init__(self, assets_dir, on_start, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = assets_dir
        self.on_start = on_start
        self.fonts = fonts or {}

        # --- 배경 ---
        self.bg = QLabel(self)
        self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "welcome_bg.png")

        # --- Start 버튼 ---
        self.btn = QPushButton(self)
        self.pm_btn         = self._load_pix(self.assets / "icons" / "start_now.png")
        self.pm_btn_pressed = self._load_pix(self.assets / "icons" / "start_now_pressed.png")

        if not self.pm_btn.isNull():
            self.btn.setIcon(QIcon(self.pm_btn))
            self.btn.setIconSize(QSize(self.BTN_W, self.BTN_H))
        else:
            self.btn.setText("Start Now")

        self.btn.setFixedSize(self.BTN_W, self.BTN_H)
        self.btn.setStyleSheet("QPushButton{border:none;background:transparent}")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.clicked.connect(on_start)

        # 클릭 시각효과(살짝 축소)
        self._btn_scale_anim = QPropertyAnimation(self.btn, b"iconSize", self)
        self._btn_scale_anim.setDuration(90)
        self._btn_scale_anim.setEasingCurve(QEasingCurve.OutCubic)

        # 눌림/해제: 아이콘 교체 + 위치 미세 이동
        self.btn.pressed.connect(self._press_fx)
        self.btn.released.connect(self._release_fx)

        self._relayout()

    # --- 헬퍼 ---
    def _load_pix(self, path):
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W * 2 or pm.height() >= self.BASE_H * 2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm, box):
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0, h0 = int(pm.width() / dpr), int(pm.height() / dpr)
        bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)): return box
        s = min(bw / w0, bh / h0)
        w, h = int(w0 * s), int(h0 * s)
        return QRect(box.x() + (bw - w) // 2, box.y() + (bh - h) // 2, w, h)

    # --- 레이아웃 ---
    def _relayout(self):
        full = self.rect()
        self.bg.setGeometry(full)
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)

        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            scaled = self.pm_bg.toImage().scaled(int(fit.width() * dpr), int(fit.height() * dpr),
                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(scaled); pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)

        bx = fit.x() + (fit.width() - self.BTN_W) // 2
        by = fit.y() + fit.height() - self.BTN_BOTTOM_MARGIN - self.BTN_H
        self.btn.setGeometry(bx, by, self.BTN_W, self.BTN_H)
        self._btn_base_pos = (bx, by)  # 눌림 복귀용

    def resizeEvent(self, e):
        self._relayout()
        super().resizeEvent(e)

    # --- 눌림/해제 효과 ---
    def _press_fx(self):
        w, h = self.BTN_W, self.BTN_H
        self._btn_scale_anim.stop()
        self._btn_scale_anim.setStartValue(QSize(w, h))
        self._btn_scale_anim.setEndValue(QSize(int(w * 0.94), int(h * 0.94)))
        self._btn_scale_anim.start()
        if not self.pm_btn_pressed.isNull():
            self.btn.setIcon(QIcon(self.pm_btn_pressed))
        if hasattr(self, "_btn_base_pos"):
            self.btn.move(self._btn_base_pos[0], self._btn_base_pos[1] + 2)

    def _release_fx(self):
        w, h = self.BTN_W, self.BTN_H
        self._btn_scale_anim.stop()
        self._btn_scale_anim.setStartValue(self.btn.iconSize())
        self._btn_scale_anim.setEndValue(QSize(w, h))
        self._btn_scale_anim.start()
        if not self.pm_btn.isNull():
            self.btn.setIcon(QIcon(self.pm_btn))
        if hasattr(self, "_btn_base_pos"):
            self.btn.move(*self._btn_base_pos)