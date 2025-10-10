# -*- coding: utf-8 -*-
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget, QLabel

class DrivingLockPage(QWidget):
    """주행 중 잠금 화면: 경고 배경만 전체 표시"""
    def __init__(self, assets_dir: Path):
        super().__init__()
        self.assets = Path(assets_dir)
        self.bg = QLabel(self)
        self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = QPixmap(str(self.assets / "bg" / "warning_bg.png"))

    def _apply(self):
        r = self.rect()
        if not self.pm_bg.isNull():
            self.bg.setGeometry(r)
            self.bg.setPixmap(
                self.pm_bg.scaled(r.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            )

    def showEvent(self, e):
        super().showEvent(e); self._apply()

    def resizeEvent(self, e):
        super().resizeEvent(e); self._apply()
