# -*- coding: utf-8 -*-
import sys, io, itertools, argparse
from pathlib import Path
from typing import Optional
import subprocess
from core.tts import speak_stream

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QRect, QSize
)
from PySide6.QtGui import QPixmap, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGraphicsOpacityEffect, QPushButton
)

# === Assets ===
ASSETS = Path(__file__).resolve().parent / "assets"
BG_PATH   = ASSETS / "bg"    / "voice_bg.png"
MIC_PATH  = ASSETS / "icons" / "voice.png"


def _pm(p: Path) -> QPixmap:
    return QPixmap(str(p))


# === Audio thread ===
class SpeakerThread(QThread):
    finished = Signal()

    def __init__(self, text: str, lang: str = "ko"):
        super().__init__()
        self.text = text
        self.lang = lang

    def run(self):
        try:
            speak_stream(self.text, self.lang)
        except Exception as e:
            print(f"[voice] audio error: {e}")
        finally:
            self.finished.emit()


class VoicePage(QWidget):
    """
    SOSPage / RecognitionPage와 같은 800x480 디자인 좌표계를 그대로 쓰는 Voice 페이지
    """
    BASE_W, BASE_H = 800, 480
    playback_finished = Signal()

    def __init__(
        self,
        assets_dir: Path,
        *,
        fonts: Optional[dict] = None,
        sign_engine=None,
        default_text: str = "안녕하세요. 음성 안내 모드가 실행됩니다.",
        on_home=None, on_recog=None, on_sos=None, on_nav=None,
        **_unused,
    ):
        super().__init__()
        self.assets = Path(assets_dir)
        self.sign_engine = sign_engine
        self.fonts = fonts or {}
        self.default_text = default_text
        self.on_home_cb  = on_home  or (lambda: None)
        self.on_recog_cb = on_recog or (lambda: None)
        self.on_nav_cb   = on_nav or (lambda: None)
        self.on_sos_cb   = on_sos   or (lambda: None)

        # 위치 파라미터
        self.LOAD_Y_PCT = 0.70
        self.LOAD_SIZE  = 62
        self.CAMERA_HINT_PT = 32
        self.VOICE_HINT_PT  = 32
        self.BOTTOM_HINT_Y_PCT = 0.88
        self.BOTTOM_HINT_H     = 85
        self.BOTTOM_HINT_MARGIN_X = 18

        self._freeze_frames = False
        self._last_frame_pm = None
        self._played_this_show = False
        self._th: SpeakerThread | None = None
        self._mode = "camera"
        self._final_text: str = ""
        self._pending_tts: bool = False

        # 배경 레이블
        self.bg_label = QLabel(self)
        self.bg_label.setAlignment(Qt.AlignCenter)

        # 카메라 레이블
        self.camera_label = QLabel(self)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)

        # 음성 아이콘 / 로딩
        self.mic_label  = QLabel(self); self.mic_label.setAlignment(Qt.AlignCenter)
        self.load_label = QLabel(self); self.load_label.setAlignment(Qt.AlignCenter)

        # 레이아웃
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        lay.addWidget(self.mic_label, 0, Qt.AlignHCenter)
        lay.addWidget(self.load_label, 0, Qt.AlignHCenter)
        lay.addStretch(1)
        self.setLayout(lay)
        # 오버레이로 빼기
        lay.removeWidget(self.load_label); self.load_label.setParent(self); self.load_label.show()
        lay.removeWidget(self.mic_label);  self.mic_label.setParent(self);  self.mic_label.show()

        self.MIC_Y_PCT = 0.42

        # 에셋 로드
        self._bg  = _pm(self.assets / "bg"    / "voice_bg.png") if (self.assets / "bg" / "voice_bg.png").exists() else _pm(BG_PATH)
        self._mic = _pm(self.assets / "icons" / "voice.png")    if (self.assets / "icons" / "voice.png").exists() else _pm(MIC_PATH)
        loads = [
            self.assets / "icons" / "voice_load1.png",
            self.assets / "icons" / "voice_load2.png",
            self.assets / "icons" / "voice_load3.png",
        ]
        self._loads = [ _pm(p if p.exists() else ASSETS / "icons" / p.name) for p in loads ]
        self._cycle = itertools.cycle(self._loads) if self._loads else None

        # 로더 타이머
        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(500)

        # 초기 상태
        self._set_voice_ui_visible(False)
        self.camera_label.show()
        self.camera_label.raise_()

        # 전환 애니메이션
        self._opacity = QGraphicsOpacityEffect(self.camera_label)
        self.camera_label.setGraphicsEffect(self._opacity)
        self._geom_anim = QPropertyAnimation(self.camera_label, b"geometry", self)
        self._opacity_anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim_group = QParallelAnimationGroup(self)
        self._anim_group.addAnimation(self._geom_anim)
        self._anim_group.addAnimation(self._opacity_anim)
        self._geom_anim.setDuration(450)
        self._opacity_anim.setDuration(450)
        self._geom_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._opacity_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim_group.finished.connect(self._on_camera_anim_finished)

        # 하단 힌트
        self.bottom_hint = QLabel("수화 입력 대기 중 …", self)
        self.bottom_hint.setAlignment(Qt.AlignCenter)
        self.bottom_hint.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-weight: 600;
                background: rgba(0,0,0,0.80);
                border-radius: 28px;
                padding: 10px 18px;
            }
        """)
        f = self.bottom_hint.font(); f.setPointSize(28); self.bottom_hint.setFont(f)
        self.bottom_hint.show()
        self._hint_opacity = QGraphicsOpacityEffect(self.bottom_hint)
        self.bottom_hint.setGraphicsEffect(self._hint_opacity)

        # 상단 아이콘들 (sos.py / recognition.py와 동일)
        def _mk_top_icon(fname: str, cb):
            b = QPushButton(self)
            pm = QPixmap(str(self.assets / "icons" / fname))
            if not pm.isNull():
                b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent;padding:0px;")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            return b

        self.btn_home  = _mk_top_icon("home.png", self.on_recog_cb)
        self.btn_voice = _mk_top_icon("voice_over_b.png", lambda: self.play(self._final_text or self.default_text))
        self.btn_nav   = _mk_top_icon("nav.png", self.on_nav_cb)
        self.btn_sos   = _mk_top_icon("sos.png", self.on_sos_cb)

        # 제스처 엔진
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self._on_hangul_progress)
            self.sign_engine.hangul_input_finished.connect(self.on_hangul_final)

    # === 공통 유틸 (recognition.py랑 맞추기) ===
    def _fit_rect_for_pixmap(self, pm: QPixmap, box: QRect) -> QRect:
        """box 안에 pm 비율 유지해서 최대한 맞춰 넣기 (위아래 안 짤리게)"""
        if pm.isNull():
            return box
        bw, bh = box.width(), box.height()
        if bw == 0 or bh == 0:
            return box
        w0, h0 = pm.width(), pm.height()
        s = min(bw / w0, bh / h0)
        w, h = int(w0 * s), int(h0 * s)
        x = box.x() + (bw - w) // 2
        y = box.y() + (bh - h) // 2
        return QRect(x, y, w, h)

    def _map_from_design(self, x, y, w, h, out_rect: QRect) -> QRect:
        """sos.py와 같은 방식: 800x480 기준 좌표를 실제 창 크기에 매핑"""
        sx = out_rect.width()  / self.BASE_W
        sy = out_rect.height() / self.BASE_H
        X = int(round(x * sx))
        Y = int(round(y * sy))
        W = int(round(w * sx))
        H = int(round(h * sy))
        return QRect(X, Y, W, H)

    # === 상태 전환 ===
    @Slot()
    def on_end_gesture(self):
        if self._mode != "camera":
            return

        self._freeze_frames = True
        try:
            if self._last_frame_pm:
                scaled = self._last_frame_pm.scaled(
                    self.camera_label.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.FastTransformation
                )
                self.camera_label.setPixmap(scaled)
        except Exception:
            pass

        if (not self._final_text) and self.sign_engine and hasattr(self.sign_engine, "get_hangul_result"):
            try:
                val = self.sign_engine.get_hangul_result() or ""
                self._final_text = val.strip()
            except Exception as e:
                print("[VoicePage] get_hangul_result failed:", e)

        self._mode = "transition"
        full = self.rect()
        target_w, target_h = int(full.width()*0.35), int(full.height()*0.20)
        tx = (full.width()-target_w)//2
        ty = int(full.height()*0.28)

        self._geom_anim.stop(); self._opacity_anim.stop()
        self._geom_anim.setStartValue(self.camera_label.geometry())
        self._geom_anim.setEndValue(QRect(tx, ty, target_w, target_h))
        self._opacity_anim.setStartValue(1.0)
        self._opacity_anim.setEndValue(0.0)
        self._anim_group.start()

    def _on_camera_anim_finished(self):
        self.camera_label.hide()
        self._set_voice_ui_visible(True)
        self._mode = "voice"

        self._hint_opacity.setOpacity(1.0)
        f = self.bottom_hint.font()
        f.setPointSize(self.VOICE_HINT_PT)
        f.setBold(False)
        self.bottom_hint.setFont(f)
        self.set_bottom_text(self._final_text or "인식된 문장을 음성으로 안내합니다")

        if self._final_text:
            if self._pending_tts:
                self.play(self._final_text)
                self._pending_tts = False
            elif not self._th or not self._th.isRunning():
                self.play(self._final_text)
        else:
            print("[VoicePage] no final text yet; waiting for hangul_input_finished")

    def _set_voice_ui_visible(self, on: bool):
        self.mic_label.setVisible(on)
        self.load_label.setVisible(on)

    # === 엔진 텍스트 ===
    @Slot(str)
    def _on_hangul_progress(self, text: str):
        if self._mode == "voice":
            return
        self.set_bottom_text(text or "수화를 시작해 주세요")

    @Slot(str)
    def on_hangul_final(self, text: str):
        self._final_text = (text or "").strip()
        self.set_bottom_text(self._final_text or "인식된 문장이 없어요")

        if self._mode != "voice":
            self._pending_tts = True
            return

        if self._final_text:
            self.play(self._final_text)

    # === 카메라 프레임 ===
    @Slot(object)
    def set_camera_image(self, frame_qimage):
        if self._mode not in ("camera", "transition") or self._freeze_frames:
            return
        try:
            if frame_qimage is None or frame_qimage.isNull():
                return
            pm = QPixmap.fromImage(frame_qimage)
            self._last_frame_pm = pm

            app = QApplication.instance().activeWindow()
            fast = Qt.FastTransformation if getattr(app, "_transitioning", False) else Qt.SmoothTransformation

            self.camera_label.setGeometry(self.rect())
            scaled = pm.scaled(self.camera_label.size(), Qt.KeepAspectRatioByExpanding, fast)
            self.camera_label.setPixmap(scaled)

            self.camera_label.raise_()
            self.bottom_hint.raise_()
            for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
                b.raise_()
        except Exception as e:
            print(f"[VoicePage] set_camera_image error: {e}")

    # === 라이프사이클 ===
    def showEvent(self, e):
        super().showEvent(e)
        self._mode = "camera"
        self._freeze_frames = False
        self._last_frame_pm = None
        self._played_this_show = False

        self._apply_bg()  # 배경 먼저
        self.camera_label.setGeometry(self.rect())
        self._opacity.setOpacity(1.0)
        self.camera_label.show()
        self._set_voice_ui_visible(False)
        self.camera_label.raise_()
        self.bottom_hint.raise_()
        self._final_text = ""
        self._pending_tts = False

        f = self.bottom_hint.font()
        f.setPointSize(self.CAMERA_HINT_PT)
        f.setBold(True)
        self.bottom_hint.setFont(f)
        self.bottom_hint.show()
        self._hint_opacity.setOpacity(1.0)

        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
            b.raise_()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_bg()

        # 마이크 위치/크기
        s = max(160, int(min(self.width(), self.height()) * 0.36))
        mx = (self.width() - s) // 2
        my = int(self.height() * getattr(self, "MIC_Y_PCT", 0.38)) - (s // 2)
        self.mic_label.setGeometry(mx, my, s, s)
        if not self._mic.isNull():
            self.mic_label.setPixmap(
                self._mic.scaled(s, s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        self.mic_label.raise_()

        # 카메라
        if self._mode in ("camera", "transition"):
            self.camera_label.setGeometry(self.rect())

        # 하단 배너
        r = self.rect()
        h = getattr(self, "BOTTOM_HINT_H", 68)
        mx2 = getattr(self, "BOTTOM_HINT_MARGIN_X", 18)
        y_pct = getattr(self, "BOTTOM_HINT_Y_PCT", 0.86)
        bar_w = r.width() - mx2*2
        bar_x = mx2
        bar_y = int(r.height() * y_pct) - (h // 2)
        self.bottom_hint.setGeometry(bar_x, bar_y, bar_w, h)
        self.bottom_hint.raise_()

        # 로딩 아이콘
        w = h = getattr(self, "LOAD_SIZE", 64)
        lx = (self.width() - w) // 2
        ly = int(self.height() * getattr(self, "LOAD_Y_PCT", 0.72)) - (h // 2)
        self.load_label.setGeometry(lx, ly, w, h)
        self.load_label.raise_()

        # 상단 아이콘 (sos.py와 동일 좌표)
        full = self.rect()
        for btn, x, y0, w0, h0 in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            rbtn = self._map_from_design(x, y0, w0, h0, out_rect=full)
            btn.setFixedSize(rbtn.width(), rbtn.height())
            btn.setIconSize(QSize(rbtn.width(), rbtn.height()))
            btn.move(rbtn.x(), rbtn.y())

        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
            b.raise_()

    def closeEvent(self, e):
        try:
            if self._th and self._th.isRunning():
                self._th.requestInterruption()
                self._th.wait(500)
        except Exception:
            pass
        super().closeEvent(e)

    # === 배경/아이콘 적용 ===
    def _apply_bg(self):
        """recognition.py와 비슷하게 배경을 박스 안에 비율 유지해서 넣기"""
        full = self.rect()
        self.bg_label.setGeometry(full)  # 기본은 전체
        if not self._bg.isNull():
            fit = self._fit_rect_for_pixmap(self._bg, full)
            pm = self._bg.scaled(fit.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.bg_label.setGeometry(fit)
            self.bg_label.setPixmap(pm)
            self.bg_label.lower()

    def _apply_pixmaps(self):
        # mic만 여기서 처리
        if not self._mic.isNull():
            sz = self.mic_label.size()
            self.mic_label.setPixmap(
                self._mic.scaled(sz, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def _tick(self):
        if not self._cycle or not self.load_label.isVisible():
            return
        pix = next(self._cycle)
        if not pix.isNull():
            self.load_label.setPixmap(
                pix.scaled(self.LOAD_SIZE, self.LOAD_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )

    def set_bottom_text(self, text: str):
        if self._mode == "voice":
            return
        self.bottom_hint.setText(text if text else "수화를 시작해 주세요")
        self.bottom_hint.raise_()
        for b in (self.btn_home, self.btn_voice, self.btn_nav, self.btn_sos):
            b.raise_()

    @Slot()
    def autoplay_once(self):
        if self._played_this_show:
            return
        self._played_this_show = True

    def play(self, text: str, lang: str = "ko", **_ignore):
        if not text or not text.strip():
            print("[VoicePage] play() skipped: empty text")
            self.playback_finished.emit()
            return
        if self._th and self._th.isRunning():
            print("[VoicePage] play() skipped: already running")
            return
        print("[VoicePage] play():", text)
        self._th = SpeakerThread(text, lang=lang)
        self._th.finished.connect(self.playback_finished.emit)
        self._th.start()


# === CLI ===
def speak_and_show_ui(text: str, ui_message: Optional[str] = None):
    app = QApplication.instance() or QApplication(sys.argv)
    page = VoicePage(ASSETS, default_text=text)
    page.show()
    sys.exit(app.exec())


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", type=str, default="안녕하세요. 음성 안내를 시작합니다.")
    ap.add_argument("--caption", type=str, default=None, help="UI에 표시할 안내 텍스트")
    args = ap.parse_args()
    speak_and_show_ui(args.say, ui_message=args.caption)


if __name__ == "__main__":
    _main()

__all__ = ["SpeakerThread", "VoicePage", "speak_and_show_ui"]
