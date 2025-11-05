# ui/chat.py
import os
from PySide6.QtCore import Qt, QRect, QTimer, QEasingCurve, QPropertyAnimation
from PySide6.QtGui import QPixmap, QPainter, QPaintEvent, QFont
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QFrame, QGraphicsOpacityEffect
from typing import Tuple

_DEBUG_NINE_SLICE = False
CAPS_OVERRIDE = {
    "user_bubble.png": {"cap_at_1x": 12},
    "com_bubble.png":  {"cap_at_1x": 12},
}

_PIXMAP_CACHE = {}
def _pm(path: str) -> QPixmap:
    pm = _PIXMAP_CACHE.get(path)
    if pm is None:
        pm = QPixmap(path)
        _PIXMAP_CACHE[path] = pm
    return pm

def _caps_for(pm: QPixmap, path: str) -> Tuple[int,int,int,int]: 
    w, h = pm.width(), pm.height()
    if w <= 0 or h <= 0: return (10, 10, 10, 10)
    name = os.path.basename(path)
    ov = CAPS_OVERRIDE.get(name, {})
    scale = max(1, round(h / 39))
    if "cap_at_1x" in ov:
        cap = int(ov["cap_at_1x"]) * scale
        t = r = b = l = cap
    else:
        base = max(6, h // 2 - 1)
        t = r = b = l = base
    if w - l - r < 2:
        cut = (2 - (w - l - r) + 1) // 2
        l = max(1, l - cut); r = max(1, r - cut)
    if h - t - b < 2:
        cut = (2 - (h - t - b) + 1) // 2
        t = max(1, t - cut); b = max(1, b - cut)
    return (t, r, b, l)

def _draw_nine_slice(p: QPainter, pm: QPixmap, dst: QRect, t: int, r: int, b: int, l: int):
    sw, sh = pm.width(), pm.height()
    cw, ch = sw - l - r, sh - t - b
    if sw <= 0 or sh <= 0 or dst.width() <= 0 or dst.height() <= 0: return
    if cw <= 0 or ch <= 0:
        p.drawPixmap(dst, pm, QRect(0, 0, sw, sh)); return
    s_tl=QRect(0,0,l,t); s_tc=QRect(l,0,cw,t); s_tr=QRect(sw-r,0,r,t)
    s_ml=QRect(0,t,l,ch); s_mc=QRect(l,t,cw,ch); s_mr=QRect(sw-r,t,r,ch)
    s_bl=QRect(0,sh-b,l,b); s_bc=QRect(l,sh-b,cw,b); s_br=QRect(sw-r,sh-b,r,b)
    dx,dy,dw,dh=dst.x(),dst.y(),dst.width(),dst.height()
    d_tl=QRect(dx,dy,l,t); d_tc=QRect(dx+l,dy,max(0,dw-l-r),t); d_tr=QRect(dx+dw-r,dy,r,t)
    d_ml=QRect(dx,dy+t,l,max(0,dh-t-b)); d_mc=QRect(dx+l,dy+t,max(0,dw-l-r),max(0,dh-t-b)); d_mr=QRect(dx+dw-r,dy+t,r,max(0,dh-t-b))
    d_bl=QRect(dx,dy+dh-b,l,b); d_bc=QRect(dx+l,dy+dh-b,max(0,dw-l-r),b); d_br=QRect(dx+dw-r,dy+dh-b,r,b)
    p.drawPixmap(d_tl,pm,s_tl); p.drawPixmap(d_tc,pm,s_tc); p.drawPixmap(d_tr,pm,s_tr)
    p.drawPixmap(d_ml,pm,s_ml); p.drawPixmap(d_mc,pm,s_mc); p.drawPixmap(d_mr,pm,s_mr)
    p.drawPixmap(d_bl,pm,s_bl); p.drawPixmap(d_bc,pm,s_bc); p.drawPixmap(d_br,pm,s_br)

class ChatBubble(QWidget):
    def __init__(self, text: str, bubble_png: str, slices=None, max_width=420, parent=None):
        super().__init__(parent)
        self._png_path = bubble_png
        self._pix = _pm(bubble_png)
        self._slices = slices or _caps_for(self._pix, bubble_png)
        self._pad_l = 12
        self._pad_r = 12
        self._pad_v_single = 1
        self._pad_v_multi  = 2
        self.is_bot = ("com_bubble" in os.path.basename(bubble_png).lower())
        
        self.TEXT_BIAS_USER = 0   # 유저 버블: 텍스트를 오른쪽으로 밀고 싶을 때 (좌패딩 +10)
        self.TEXT_BIAS_BOT  = 0    # 봇은 필요 없으면 0
        self.bg_opacity = 0.45 if self.is_bot else 1.0 
        self.COM_DELAY_MS = 2000

        self.setAttribute(Qt.WA_TranslucentBackground, True)

        # 레이아웃
        lay = QVBoxLayout(self)
        lay.setContentsMargins(self._pad_l, self._pad_v_single, self._pad_r, self._pad_v_single)

        # ----- 라벨: 반응형 폰트 + 긴단어 소프트랩 -----
        self.lbl = QLabel(self._soft_wrap_text(text), self)
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self._apply_auto_font()  # px 고정 대신 pt 기반으로 반응형
        self.lbl.setStyleSheet("color:#FFFFFF;background:transparent;")
        lay.addWidget(self.lbl)

        # 최소 크기(9-slice 캡 + 내부 패딩 고려)
        t, r, b, l = self._slices
        min_w_caps = l + r + 2
        min_h_caps = t + b + 2
        self.setMinimumWidth(min_w_caps + (self._pad_l + self._pad_r))
        self.setMinimumHeight(min_h_caps)
        self.setMaximumWidth(max_width)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self._apply_wrap_metrics()

    # -------- helpers --------
    def _apply_auto_font(self):
        """DPI 기반 pt 폰트 크기(반응형)."""
        f = QFont(self.lbl.font())
        # 대략 14~18pt 범위 안에서 DPI에 비례
        pt = max(20, min(28, int(self.logicalDpiY() / 5)))
        f.setPointSize(pt)
        self.lbl.setFont(f)

    def _soft_wrap_text(self, s: str) -> str:
        """
        공백 없는 매우 긴 토큰(예: URL, 해시)이 버블 밖으로 새지 않도록
        20~24자마다 zero-width space(\\u200b)를 삽입.
        """
        import re
        tokens = s.split(" ")
        out = []
        for tok in tokens:
            if len(tok) <= 24:
                out.append(tok)
                continue
            # 영문/숫자 연속 토큰만 분절(한글은 보통 자동 줄바꿈 처리됨)
            if re.fullmatch(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", tok):
                chunks = [tok[i:i+24] for i in range(0, len(tok), 24)]
                out.append("\u200b".join(chunks))
            else:
                out.append(tok)
        return " ".join(out)

    # -------- layout metrics --------
    def _apply_wrap_metrics(self):
        fm = self.lbl.fontMetrics()
        line_h = fm.lineSpacing()

        # 폰트 기준으로 패딩 스케일 (반응형 내부 여백)
        self._pad_l = self._pad_r = min(34, max(14, line_h * 0.5))
        self._pad_v_single = min(16, max(4, line_h * 0.45))
        self._pad_v_multi  = min(20, max(6, line_h * 0.55))
        
        if not self.is_bot:
            self._pad_l += getattr(self, "TEXT_BIAS_USER", 0)
        else:
            self._pad_r += getattr(self, "TEXT_BIAS_BOT", 0)

        avail = max(10, (self.width() or self.maximumWidth() or 240) - (self._pad_l + self._pad_r))
        br = fm.boundingRect(0, 0, avail, 10**6, Qt.TextWordWrap, self.lbl.text())
        lines = max(1, (br.height() + line_h - 1) // line_h)
        pad_v = self._pad_v_single if lines == 1 else self._pad_v_multi
        self.layout().setContentsMargins(self._pad_l, pad_v, self._pad_r, pad_v)

        # 최소 폭 재보정(패딩 포함) — 레이아웃이 아주 좁아질 때도 안전
        t, r, b, l = self._slices
        min_w_caps = l + r + 2
        self.setMinimumWidth(min_w_caps + (self._pad_l + self._pad_r))

    # -------- public API --------
    def set_fixed_width(self, w: int):
        self.setFixedWidth(int(w))
        self._apply_wrap_metrics()
        self.adjustSize()
        self.updateGeometry()

    # (선택) 텍스트 교체 시도용 메서드
    def set_text(self, text: str):
        self.lbl.setText(self._soft_wrap_text(text or ""))
        self._apply_auto_font()
        self._apply_wrap_metrics()
        self.update()

    # -------- Qt events --------
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._apply_all()
        QTimer.singleShot(0, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def paintEvent(self, e: QPaintEvent):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if not self._pix.isNull():
            if self.bg_opacity < 1.0 and self.is_bot:
                p.setOpacity(self.bg_opacity)   # ★ com_bubble만 투명도 적용
            t, r, b, l = self._slices
            _draw_nine_slice(p, self._pix, self.rect(), t, r, b, l)


class ChatPanel(QWidget):
    def __init__(self, user_png: str, bot_png: str, parent=None):
        super().__init__(parent)
        self.user_png, self.bot_png = user_png, bot_png
        self.user_slices = _caps_for(_pm(user_png), user_png); self.bot_slices  = _caps_for(_pm(bot_png),  bot_png)
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        self.scroll=QScrollArea(self); 
        self.scroll.setWidgetResizable(True); 
        self.scroll.setFrameShape(QFrame.NoFrame)
        # 스크롤바 숨김 + 패딩 제거
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setStyleSheet("""
        QScrollArea { border: none; padding: 0; margin: 0; }
        QScrollArea > QWidget > QWidget { margin: 0; }   /* viewport 내부 위젯 */
        QScrollBar:vertical { width: 0px; margin: 0; background: transparent; }
        QScrollBar::handle:vertical,
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical { height: 0px; margin: 0; border: none; background: transparent; }
        """)
        root.addWidget(self.scroll)
        self.container=QWidget(); 
        self.v=QVBoxLayout(self.container); 
        self.v.setSpacing(8)
        self.v.setContentsMargins(15,8,4,8); 
        
        self.v.insertStretch(0, 1)
        
        self.scroll.setWidget(self.container)
        self.setStyleSheet("QScrollArea{background:transparent;} QWidget{background:transparent;}")
    
    def _smooth_scroll_to_bottom(self, dur=240):
        sb = self.scroll.verticalScrollBar()
        if not sb: return
        start = sb.value()
        end = sb.maximum()
        if end <= start:
            return
        # 애니메이션이 GC로 사라지지 않도록 참조 유지
        if not hasattr(self, "_scroll_anim"):
            self._scroll_anim = None
        anim = QPropertyAnimation(sb, b"value", self)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setDuration(dur)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        # 이전 애니메이션 중지
        if self._scroll_anim and self._scroll_anim.state() == QPropertyAnimation.Running:
            self._scroll_anim.stop()
        self._scroll_anim = anim
        anim.start()
    
    def _single_line_pixels(self, bubble: QWidget, text: str) -> int:
        fm = bubble.lbl.fontMetrics()
        # 이모지/한글 오차 보정: boundingRect와 horizontalAdvance 둘 중 큰 값 사용
        w1 = fm.horizontalAdvance(text)
        w2 = fm.boundingRect(text).width()
        text_px = max(w1, w2)
        return text_px + bubble._pad_l + bubble._pad_r + 2  # 여유 2px

    def append(self, text: str, role: str = "user"):
        text = (text or "").strip()
        if not text:
            return

        if role == "com":
            # ★ com은 살짝 딜레이 후 실제 추가
            QTimer.singleShot(getattr(self, "COM_DELAY_MS", 2000), 
                            lambda: self._append_row(text, role))
        else:
            # user는 즉시 추가
            self._append_row(text, role)

    def _append_row(self, text: str, role: str):
        # 1) 행 레이아웃 구성
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        bubble = ChatBubble(
            text,
            self.user_png if role == "user" else self.bot_png,
            self.user_slices if role == "user" else self.bot_slices,
            max_width=self._bounds()[1]
        )

        # 2) 한 줄이면 wrap 끄고 폭 결정
        min_w, max_w = self._bounds()
        fm = bubble.lbl.fontMetrics()
        single_px = max(fm.horizontalAdvance(text), fm.boundingRect(text).width()) + bubble._pad_l + bubble._pad_r + 2
        force_single = single_px <= max_w
        bubble.lbl.setWordWrap(not force_single)  # == force_single ? False : True
        bubble.lbl.setWordWrap(not force_single)
        target_w = max(min_w, min(max_w, single_px))
        bubble.set_fixed_width(target_w)

        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0, alignment=Qt.AlignRight)
        else:
            row_layout.addWidget(bubble, 0, alignment=Qt.AlignLeft)
            row_layout.addStretch(1)

        # 3) 행 컨테이너(펼치기/페이드 애니메이션용)
        row_container = QWidget(self.container)
        row_container.setAttribute(Qt.WA_TranslucentBackground, True)
        row_container.setLayout(row_layout)
        row_container.setMaximumHeight(1)  # 접힌 상태로 시작

        eff = QGraphicsOpacityEffect(row_container)
        row_container.setGraphicsEffect(eff)
        eff.setOpacity(0.0)

        # 아래 정렬 유지: 맨 끝에 추가
        self.v.addWidget(row_container)

        # 4) 실제 높이 측정 후 애니메이션
        row_container.adjustSize()
        target_h = max(1, row_container.sizeHint().height())

        grow = QPropertyAnimation(row_container, b"maximumHeight", self)
        grow.setStartValue(1)
        grow.setEndValue(target_h)
        grow.setDuration(250)
        grow.setEasingCurve(QEasingCurve.OutCubic)

        fade = QPropertyAnimation(eff, b"opacity", self)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.setDuration(220)
        fade.setEasingCurve(QEasingCurve.OutCubic)

        def _after():
            self._apply_all()
            self._smooth_scroll_to_bottom(dur=240)

        grow.finished.connect(_after)
        # 참조 유지(가비지 방지)
        if not hasattr(self, "_row_anims"):
            self._row_anims = []
        self._row_anims.append((grow, fade))
        grow.start(); fade.start()

    def clear(self):
        for idx in range(self.v.count()-1, 0, -1):
            item = self.v.takeAt(idx)
            if not item:
                continue
            lay = item.layout()
            if lay:
                while lay.count():
                    w = lay.takeAt(0).widget()
                    if w:
                        w.setParent(None); w.deleteLater()
            else:
                w = item.widget()
                if w:
                    w.setParent(None); w.deleteLater()

                
    def _bounds(self) -> Tuple[int, int]:
        vw = self.scroll.viewport().width() or self.width() or 400
        min_w = max(90, int(vw * 0.05))     # 화면의 5%
        max_w = max(min_w+60, int(vw * 0.94))  # 화면의 94%
        return min_w, max_w
    
    def _calc_width(self, bubble: QWidget, text: str) -> int:
        min_w, max_w = self._bounds()
        fm = bubble.lbl.fontMetrics()
        single_px = fm.horizontalAdvance(text) + bubble._pad_l + bubble._pad_r
        if single_px > max_w:
            return max_w
        return max(min_w, single_px)

    
    def _apply_all(self):
        for i in range(self.v.count()):
            item = self.v.itemAt(i)
            if not item: continue
            lay = item.layout();
            if not lay: continue
            for j in range(lay.count()):
                w = lay.itemAt(j).widget()
                if isinstance(w, ChatBubble): w.set_fixed_width(self._calc_width(w, w.lbl.text()))
    def resizeEvent(self, e): super().resizeEvent(e); self._apply_all()