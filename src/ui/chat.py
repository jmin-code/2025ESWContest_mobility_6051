# ui/chat.py
import os
from PySide6.QtCore import Qt, QRect, QTimer
from PySide6.QtGui import QPixmap, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea, QSizePolicy, QFrame
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

def _caps_for(pm: QPixmap, path: str) -> Tuple[int,int,int,int]: # <<< 1차 수정된 부분
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
        self._png_path = bubble_png; self._pix = _pm(bubble_png); self._slices = slices or _caps_for(self._pix, bubble_png)
        is_bot = ("com_bubble" in os.path.basename(bubble_png).lower())
        self._pad_l=12; self._pad_r=12; self._pad_v_single=1; self._pad_v_multi=2
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        lay=QVBoxLayout(self); lay.setContentsMargins(self._pad_l,self._pad_v_single,self._pad_r,self._pad_v_single)
        self.lbl=QLabel(text,self); self.lbl.setWordWrap(True); self.lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum); self.lbl.setStyleSheet("color:#FFFFFF;background:transparent;font-size:16px;")
        lay.addWidget(self.lbl)
        t,r,b,l=self._slices; self.setMinimumWidth(l+r+2); self.setMinimumHeight(t+b+2)
        self.setMaximumWidth(max_width); self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        self._apply_wrap_metrics()
    def _apply_wrap_metrics(self):
        fm=self.lbl.fontMetrics(); avail=max(10,(self.width() or self.maximumWidth() or 240)-(self._pad_l+self._pad_r))
        br=fm.boundingRect(0,0,avail,10**6,Qt.TextWordWrap,self.lbl.text())
        lines=max(1,(br.height()+fm.lineSpacing()-1)//fm.lineSpacing())
        pad_v=self._pad_v_single if lines==1 else self._pad_v_multi
        self.layout().setContentsMargins(self._pad_l,pad_v,self._pad_r,pad_v)
    def set_fixed_width(self,w:int): self.setFixedWidth(int(w));self._apply_wrap_metrics();self.adjustSize();self.updateGeometry()
    def resizeEvent(self,e): super().resizeEvent(e); self._apply_wrap_metrics()
    def paintEvent(self,e:QPaintEvent):
        p=QPainter(self); p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        if not self._pix.isNull(): t,r,b,l=self._slices; _draw_nine_slice(p,self._pix,self.rect(),t,r,b,l)

class ChatPanel(QWidget):
    def __init__(self, user_png: str, bot_png: str, parent=None):
        super().__init__(parent)
        self.user_png, self.bot_png = user_png, bot_png
        self.user_slices = _caps_for(_pm(user_png), user_png); self.bot_slices  = _caps_for(_pm(bot_png),  bot_png)
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        self.scroll=QScrollArea(self); self.scroll.setWidgetResizable(True); self.scroll.setFrameShape(QFrame.NoFrame)
        root.addWidget(self.scroll)
        self.container=QWidget(); self.v=QVBoxLayout(self.container); self.v.setSpacing(8)
        self.v.setContentsMargins(15,8,10,8); self.v.addStretch(1)
        self.scroll.setWidget(self.container)
        self.setStyleSheet("QScrollArea{background:transparent;} QWidget{background:transparent;}")
    def append(self, text: str, role: str = "user"):
        text=(text or "").strip()
        if not text: return
        row=QHBoxLayout(); row.setContentsMargins(0,0,0,0); row.setSpacing(4)
        bubble=ChatBubble(text, self.user_png if role=="user" else self.bot_png, self.user_slices if role=="user" else self.bot_slices, max_width=self._bounds()[1])
        bubble.set_fixed_width(self._calc_width(bubble, text))
        if role=="user": row.addStretch(1); row.addWidget(bubble,0,alignment=Qt.AlignRight)
        else: row.addWidget(bubble,0,alignment=Qt.AlignLeft); row.addStretch(1)
        self.v.insertLayout(self.v.count()-1, row)
        QTimer.singleShot(0, lambda: (self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()), self._apply_all()))
    def clear(self):
        while self.v.count() > 1:
            item = self.v.takeAt(0)
            if item is None: continue
            lay=item.layout()
            if lay:
                while lay.count():
                    w=lay.takeAt(0).widget()
                    if w: w.setParent(None); w.deleteLater()
            else:
                w=item.widget()
                if w: w.setParent(None); w.deleteLater()
    def _bounds(self) -> Tuple[int, int]: # <<< 2차 수정된 부분
        vw=self.scroll.viewport().width() or self.width() or 400; max_w=max(120,vw-30); min_w=50; return min_w, max_w
    def _calc_width(self,bubble:QWidget,text:str)->int:
        min_w,max_w=self._bounds(); fm=bubble.lbl.fontMetrics(); single_px=fm.horizontalAdvance(text)+bubble._pad_l+bubble._pad_r
        if single_px>max_w: return max_w
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