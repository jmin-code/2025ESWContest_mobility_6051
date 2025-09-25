# -*- coding: utf-8 -*-
import sys, io, itertools, argparse
from pathlib import Path
from typing import Optional
import subprocess
from core.tts import speak_stream

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot,
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QRect,
)
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QVBoxLayout, QGraphicsOpacityEffect
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

# === Stacked-page version (used by main.py) ===
class VoicePage(QWidget):
    """QStackedWidget 안에서 쓰는 Voice 전용 페이지."""
    BASE_W, BASE_H = 800, 480

    def __init__(
        self,
        assets_dir: Path,
        *,
        fonts: Optional[dict] = None,
        sign_engine=None,
        default_text: str = "안녕하세요. 음성 안내 모드가 실행됩니다.",
        on_home=None, on_recog=None, on_sos=None,
        **_unused,

    ):
        super().__init__()
        self.assets = Path(assets_dir)
        self.sign_engine = sign_engine
        self.fonts = fonts or {}
        self.default_text = default_text
        self.on_home_cb  = on_home  or (lambda: None)
        self.on_recog_cb = on_recog or (lambda: None)
        self.on_sos_cb   = on_sos   or (lambda: None)
        self.LOAD_Y_PCT = 0.70   # 화면 높이의 78% 지점에 로딩 점 배치 (원하면 0.70~0.85로 조절)
        self.LOAD_SIZE  = 62     # 로딩 점 아이콘 크기(px)
        self.CAMERA_HINT_PT = 26   # 카메라 화면에서 크게
        self.VOICE_HINT_PT  = 18   # 애니메이션 후(음성 UI)엔 조금 작게
        
        self._played_this_show = False
        self._th: SpeakerThread | None = None
        self._mode = "camera"  # "camera" -> "transition" -> "voice"

        # 배경
        self.bg_label = QLabel(self); self.bg_label.setAlignment(Qt.AlignCenter)

        # 카메라(풀스크린 확인용)
        self.camera_label = QLabel(self)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)

        # 음성 아이콘 UI (초기엔 숨김)
        self.mic_label  = QLabel(self); self.mic_label.setAlignment(Qt.AlignCenter)
        self.load_label = QLabel(self); self.load_label.setAlignment(Qt.AlignCenter)
        # self.text_label = QLabel("음성을 출력 중입니다...", self)
        # self.text_label.setAlignment(Qt.AlignCenter)
        # self.text_label.setStyleSheet("color: white;")
        # self.text_label.setFont(QFont("Arial", 28))
        
        self._final_text: str = ""
        self._pending_tts: bool = False


        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addStretch(1)
        lay.addWidget(self.mic_label, 0, Qt.AlignHCenter)
        lay.addWidget(self.load_label, 0, Qt.AlignHCenter)
        # lay.addWidget(self.text_label, 0, Qt.AlignHCenter)
        lay.addStretch(1)
        self.setLayout(lay)

        lay.removeWidget(self.load_label)   # 레이아웃에서 분리
        self.load_label.setParent(self)     # 페이지에 직접 부착(오버레이)
        self.load_label.show()


        # Assets
        self._bg  = _pm(self.assets / "bg"    / "voice_bg.png") if (self.assets / "bg" / "voice_bg.png").exists() else _pm(BG_PATH)
        self._mic = _pm(self.assets / "icons" / "voice.png")    if (self.assets / "icons" / "voice.png").exists() else _pm(MIC_PATH)
        loads = [
            self.assets / "icons" / "voice_load1.png",
            self.assets / "icons" / "voice_load2.png",
            self.assets / "icons" / "voice_load3.png",
        ]
        self._loads = [ _pm(p if p.exists() else ASSETS / "icons" / p.name) for p in loads ]
        self._cycle = itertools.cycle(self._loads) if self._loads else None

        self._apply_pixmaps()

        # 로더 타이머
        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(500)

        # 초기 상태: 카메라만 표시, 아이콘 UI 숨김
        self._set_voice_ui_visible(False)
        self.camera_label.show()
        self.camera_label.raise_()

        # 애니메이션 준비
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

        self.bottom_hint = QLabel("수화 입력 대기 중 …", self)
        self.bottom_hint.setAlignment(Qt.AlignCenter)
        self.bottom_hint.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-weight: 600;
                background: rgba(255,255,255,0.10);
                border-radius: 28px;
                padding: 10px 18px;
            }
        """)
        # 기본 폰트 크기 살짝 키우고, 처음엔 표시
        f = self.bottom_hint.font(); f.setPointSize(18); self.bottom_hint.setFont(f)
        self.bottom_hint.show()

        self._hint_opacity = QGraphicsOpacityEffect(self.bottom_hint)
        self.bottom_hint.setGraphicsEffect(self._hint_opacity)
        self._hint_fade = QPropertyAnimation(self._hint_opacity, b"opacity", self)
        self._hint_fade.setDuration(300)
        self._hint_fade.setStartValue(1.0)
        self._hint_fade.setEndValue(0.0)
        self._hint_fade.finished.connect(self.bottom_hint.hide)

        # 엔진 시그널 연결(프레임 갱신)
        if self.sign_engine:
            self.sign_engine.frame_updated.connect(self.set_camera_image)
            self.sign_engine.hangul_result_updated.connect(self._on_hangul_progress)   # 진행 중 텍스트 (이미 있다면 유지)
            self.sign_engine.hangul_input_finished.connect(self.on_hangul_final)       # ★ 최종 문자열


    # ---------- 상태 전환 ----------
    @Slot()
    def on_end_gesture(self):
        """end 제스처 확인 → 카메라 축소/페이드 후 음성 UI로 전환"""
        if self._mode != "camera":
            return
        # ★ 레이스 방지: 최종 텍스트 즉시 확보 시도
        if (not self._final_text) and self.sign_engine and hasattr(self.sign_engine, "get_hangul_result"):
            try:
                val = self.sign_engine.get_hangul_result() or ""
                self._final_text = val.strip()
                print("[VoicePage] pulled final text from engine:", self._final_text)
            except Exception as e:
                print("[VoicePage] get_hangul_result failed:", e)

        self._mode = "transition"
        full = self.rect()

        # 카메라 현재(풀) → 중앙 작은 박스로 축소
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
        # 카메라 숨기고 음성 UI 표시
        self.camera_label.hide()
        self._set_voice_ui_visible(True)
        self._mode = "voice"
        
        try:
            self._hint_fade.stop()
            self._hint_opacity.setOpacity(1.0)
            self._hint_fade.start()
        except Exception:
            self.bottom_hint.hide()

        # 기본 자막 업데이트
        # if hasattr(self, "set_bottom_text"):
        #     self.set_bottom_text(self._final_text or "음성 안내를 시작합니다...")

        # ★ 최종 문자열이 이미 도착했다면 그걸 재생, 아니면 대기
        if self._final_text:
            self.play(self._final_text)  # caption 없이
            self._pending_tts = False
        else:
            print("[VoicePage] no final text yet; waiting for hangul_input_finished")

    def _set_voice_ui_visible(self, on: bool):
        self.mic_label.setVisible(on)
        self.load_label.setVisible(on)
        # self.text_label.setVisible(on)

    @Slot(str)
    def _on_hangul_progress(self, text: str):
        if self._mode == "voice":
            return  
        self.set_bottom_text(text or "수화를 시작해 주세요")

    @Slot(str)
    def on_hangul_final(self, text: str):
        """최종 문자열 도착: 저장해 두고, 상황에 맞게 즉시/지연 재생."""
        self._final_text = (text or "").strip()
        if hasattr(self, "set_bottom_text"):
            self.set_bottom_text(self._final_text or "인식된 문장이 없어요")

        # 아직 카메라 모드면 나중에(전환 후) 재생
        if self._mode != "voice":
            self._pending_tts = True
            return

        # 이미 아이콘 UI 상태면 바로 재생
        if self._final_text:
            self.play(self._final_text, caption="음성으로 읽어드릴게요.")


    # ---------- 엔진 프레임 ----------
    @Slot(object)
    def set_camera_image(self, frame_qimage):
        # 카메라 모드에서만 갱신
        if self._mode not in ("camera", "transition"):
            return
        try:
            if frame_qimage is None or frame_qimage.isNull():
                return
            # 카메라 라벨은 항상 전체 창 채우기
            self.camera_label.setGeometry(self.rect())
            pm = QPixmap.fromImage(frame_qimage)
            scaled = pm.scaled(self.camera_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.camera_label.setPixmap(scaled)
            if not self.camera_label.isVisible():
                self.camera_label.show()
            self.camera_label.raise_()
            self.bottom_hint.raise_()
        except Exception as e:
            print(f"[VoicePage] set_camera_image error: {e}")

    # ---------- 라이프사이클 ----------
    def showEvent(self, e):
        super().showEvent(e)
        # 카메라 모드로 리셋
        self._mode = "camera"
        self._played_this_show = False
        self.bg_label.setGeometry(self.rect())
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
        if hasattr(self, "_hint_opacity"):
            self._hint_opacity.setOpacity(1.0)


    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.bg_label.setGeometry(self.rect())
        self._apply_pixmaps()
        if self._mode in ("camera", "transition"):
            self.camera_label.setGeometry(self.rect())

        # ── 하단 배너 위치 (좌우 18px 마진, 높이 68px) ──
        m = 18
        h = 68
        r = self.rect()
        self.bottom_hint.setGeometry(m, r.bottom() - h - m, r.width() - m*2, h)
        # 항상 맨 위로
        self.bottom_hint.raise_()
        
        # ── 로딩 점 위치/크기 동적 배치 ──
        w = h = getattr(self, "LOAD_SIZE", 64)
        x = (self.width() - w) // 2
        y = int(self.height() * getattr(self, "LOAD_Y_PCT", 0.78)) - (h // 2)
        self.load_label.setGeometry(x, y, w, h)
        self.load_label.raise_()

    
    def closeEvent(self, e):
        try:
            if self._th and self._th.isRunning():
                self._th.requestInterruption()
                self._th.wait(500)
        except Exception:
            pass
        super().closeEvent(e)

    # ---------- 음성 아이콘/로더/TTS ----------
    def _apply_pixmaps(self):
        if not self._bg.isNull():
            self.bg_label.setPixmap(self._bg.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
            self.bg_label.lower()
        if not self._mic.isNull():
            self.mic_label.setPixmap(self._mic.scaled(144, 144, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _tick(self):
        if not self._cycle or not self.load_label.isVisible():
            return
        pix = next(self._cycle)
        if not pix.isNull():
            self.load_label.setPixmap(
                pix.scaled(self.LOAD_SIZE, self.LOAD_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
    
    # VoicePage 클래스 안에 추가
    @Slot(str)
    def _on_hangul_progress(self, text: str):
        # 진행 중 한글 조합 결과가 들어옴 (예: "ㅇ인하ㅏ")
        self.set_bottom_text(text or "")

    def set_bottom_text(self, text: str):
        if self._mode == "voice":
            return
        # bottom_hint 라벨에 바로 반영
        self.bottom_hint.setText(text if text else "수화를 시작해 주세요")
        self.bottom_hint.raise_()


    @Slot()
    def autoplay_once(self):
        if self._played_this_show:
            return
        self._played_this_show = True
        # self.play(self.default_text, caption="음성을 출력 중입니다...")

    def play(self, text: str, lang: str = "ko"):
        if not text or not text.strip():
            print("[VoicePage] play() skipped: empty text")
            return
        if self._th and self._th.isRunning():
            print("[VoicePage] play() skipped: already running")
            return
        print("[VoicePage] play():", text)
        self._th = SpeakerThread(text, lang=lang)
        self._th.finished.connect(lambda: print("[VoicePage] TTS finished"))
        self._th.start()

# === CLI: run voice window standalone ===
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
