# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt, QRect, QSize, QUrl, QTimer, Slot
from PySide6.QtGui import QPixmap, QIcon, QRegion, QPainterPath
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage


# ===== 기본 좌표 (서울 시청 근처) =====
DEFAULT_LATITUDE = 37.4505    # 예: 서울 위도
DEFAULT_LONGITUDE = 126.6572  # 예: 서울 경도

class SOSPage(QWidget):
    """
    배경(sos_bg.png)에 카드가 이미 포함되어 있으므로 별도 카드 위젯을 만들지 않는다.
    배경 이미지의 카드 영역을 비율로 계산해 그 안에 지도/라벨/버튼만 배치.
    """
    BASE_W, BASE_H = 800, 480   # 앱 창 크기 기준
    # 배경 속 카드가 차지하는 화면 비율 (x, y, w, h) — 스샷 기준 튜닝값
    CARD_PCT = (0.22, 0.23, 0.56, 0.70)  # 좌우 22%, 위 23%, 폭 56%, 높이 70%
    # 지도 크기 비율(카드 기준): 가로 74%, 세로 60%)
    MAP_W_PCT = 0.74
    MAP_H_PCT = 0.60
    # 지도 상단 여백(카드 높이 대비 비율)
    MAP_TOP_PCT = 0.05
    # 버튼 크기/위치 (카드 기준 비율)
    BTN_W_PCT       = 0.28   # 가로폭: 카드의 32% (원하면 0.28~0.36 사이로 조절)
    BTN_H_PX        = 40    # 높이(픽셀)
    BTN_BOTTOM_PCT  = 0.005   # 카드 하단에서 위로 띄우는 여백 비율
    BTN_ICON_SCALE  = 0.9    # 아이콘은 버튼보다 살짝 작게



    def __init__(self, assets_dir: Path, on_home=None, on_nav=None, on_send=None):
        super().__init__()
        self.assets = Path(assets_dir)
        self.on_home = on_home or (lambda: None)
        self.on_nav  = on_nav  or (lambda: None)
        self.on_send = on_send or (lambda: None)

        # --- 배경 ---
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = QPixmap(str(self.assets / "bg" / "sos_bg.png"))

        # --- 상단 아이콘(배경 상단 바 영역용) ---
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = QPixmap(str(self.assets / "icons" / fname))
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            return b
        self.btn_home = mk_icon("home.png", self.on_home)
        self.btn_nav  = mk_icon("nav.png",  self.on_nav)
        self.btn_sos  = mk_icon("sos_b.png", lambda: None)

        # --- 카드 내부에 올릴 요소들 ---
        # self.title = QLabel("Emergency Rescue Request", self)
        # self.title.setAlignment(Qt.AlignCenter)
        # self.title.setStyleSheet("color:#EDEDED;")
        # self.title.setFont(self._font(20, True))

        self.web = QWebEngineView(self)
        self.web.setPage(QWebEnginePage(self.web))
        self.web.loadFinished.connect(self._on_map_loaded)

        self.coords_title = QLabel("Current Location Coordinates:", self)
        self.coords_title.setAlignment(Qt.AlignCenter)
        self.coords_title.setStyleSheet("color:#CFCFCF;")
        self.coords_title.setFont(self._font(13))

        self.lat_label = QLabel("Latitude: —", self)
        self.lat_label.setAlignment(Qt.AlignCenter)
        self.lat_label.setStyleSheet("color:#DADADA;")
        self.lat_label.setFont(self._font(12))

        self.lng_label = QLabel("Longitude: —", self)
        self.lng_label.setAlignment(Qt.AlignCenter)
        self.lng_label.setStyleSheet("color:#DADADA;")
        self.lng_label.setFont(self._font(12))

        self.btn_send = QPushButton(self)
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setStyleSheet("QPushButton{border:none;background:transparent}")
        pm_send = QPixmap(str(self.assets / "icons" / "send_sos.png"))
        if not pm_send.isNull():
            self.btn_send.setIcon(QIcon(pm_send))
        else:
            self.btn_send.setText("Send SOS")
            self.btn_send.setStyleSheet("QPushButton { color: white; background:#E53935; border-radius: 28px; padding:14px 28px; }")
        self.btn_send.clicked.connect(self.on_send)

        # 상태
        self._latlng: Optional[Tuple[float, float]] = None
        self._map_ready = False
        self._initial_loaded = False

        self._relayout()
        QTimer.singleShot(0, self._load_initial_map)

    # ========== 외부 API ==========
    # @Slot(float, float)
    # def set_location(self, lat: float, lng: float):
    #     self._latlng = (lat, lng)
    #     ns = "N" if lat >= 0 else "S"
    #     ew = "E" if lng >= 0 else "W"
    #     self.lat_label.setText(f"Latitude: {abs(lat):.4f}° {ns}")
    #     self.lng_label.setText(f"Longitude: {abs(lng):.4f}° {ew}")
    #     if self._map_ready:
    #         self._center_map(lat, lng)
    @Slot(float, float)
    def set_location(self, lat: float, lng: float):
        """GPS로부터 새로운 위치를 받으면 호출되는 메서드"""
        if self._latlng == (lat, lng):
            return
        
        self._latlng = (lat, lng)
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lng >= 0 else "W"
        self.lat_label.setText(f"Latitude: {abs(lat):.4f}° {ns}")
        self.lng_label.setText(f"Longitude: {abs(lng):.4f}° {ew}")

        # if self._map_ready:
        #     # 자바스크립트 함수 호출
        #     self._center_map(lat, lng, place_marker=True)
        if self._map_ready:
            # search.html의 자바스크립트 함수 호출
            self._update_map_location(lat, lng)

    # ========== 내부 로직 ==========
    def _font(self, size: int, bold: bool=False):
        f = self.font()
        f.setPointSize(size)
        f.setBold(bold)
        return f

    def _load_initial_map(self):
        if self._initial_loaded:
            return
        self._initial_loaded = True
        # lat, lng = self._latlng if self._latlng else (37.554678, 126.970609)
        # # sos.html 우선, 없으면 map.html
        # def url(base):
        #     q = QUrl(base)
        #     q.setQuery(f"cLat={lat}&cLng={lng}&mode=sos")
        #     return q
        # self.web.load(url("http://localhost:5050/sos.html"))
        self.web.load(QUrl(f"http://localhost:5050/search.html"))

    # @Slot(bool)
    # def _on_map_loaded(self, ok: bool):
    #     if not ok:
    #         lat, lng = self._latlng if self._latlng else (37.554678, 126.970609)
    #         self.web.load(QUrl(f"http://localhost:5050/map.html?cLat={lat}&cLng={lng}&mode=sos"))
    #         return
    #     self._map_ready = True
    #     lat, lng = self._latlng if self._latlng else (37.554678, 126.970609)
    #     self._center_map(lat, lng, place_marker=True)

    # def _center_map(self, lat: float, lng: float, place_marker: bool=True):
    #     js = []
    #     if place_marker:
    #         js += [
    #             f"try{{ showSOS({lat},{lng}); }}catch(e){{}}",
    #             f"try{{ centerAndMark({lat},{lng}); }}catch(e){{}}",
    #             f"try{{ setCenter({lat},{lng}); addMarker({lat},{lng}); }}catch(e){{}}",
    #         ]
    #     else:
    #         js += [
    #             f"try{{ centerMap({lat},{lng}); }}catch(e){{}}",
    #             f"try{{ setCenter({lat},{lng}); }}catch(e){{}}",
    #         ]
    #     self.web.page().runJavaScript(";".join(js))
    @Slot(bool)
    def _on_map_loaded(self, ok: bool):
        if not ok:
            print("[SOS] 맵 로드 실패!")
            return
        self._map_ready = True
        # 지도가 로드된 후, 실제 위치가 있으면 해당 위치로, 없으면 기본 위치로 중앙 설정
        lat, lng = self._latlng if self._latlng else (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self._update_map_location(lat, lng)

    def _update_map_location(self, lat: float, lng: float):
        # search.html에 새로 추가할 JS 함수를 호출
        js_code = f"updateCurrentLocation({lat}, {lng});"
        self.web.page().runJavaScript(js_code)

    def _center_map(self, lat: float, lng: float, place_marker: bool=True):
        # 범용 map.html의 JS 함수를 호출하도록 수정
        if place_marker:
            js_code = f"setCenterAndMarker({lat}, {lng});"
        else:
            js_code = f"setCenter({lat}, {lng});"
        self.web.page().runJavaScript(js_code)

    # ========== 레이아웃 ==========
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def _relayout(self):
        full = self.rect()

        # 배경
        self.bg.setGeometry(full)
        if not self.pm_bg.isNull():
            self.bg.setPixmap(self.pm_bg.scaled(full.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self.bg.setStyleSheet("background:#111")

        # 배경 속 카드 영역(비율) → 실제 좌표
        cx, cy, cw, ch = self._card_rect(full)

        pad = int(cw * 0.06)

        # ── 타이틀 제거했으므로 지도 y 시작을 좀 더 위에서 시작 ──
        y = int(ch * self.MAP_TOP_PCT)

        # ── 지도: 가로는 더 좁게, 세로는 더 길게 ──
        map_w = int(cw * self.MAP_W_PCT)
        map_h = int(ch * self.MAP_H_PCT)
        mx = cx + (cw - map_w) // 2                 # 카드 중앙 정렬
        my = cy + y
        self.web.setGeometry(mx, my, map_w, map_h)
        self._apply_round_mask(self.web, radius=22)

        # 좌표 라벨들: 지도 아래로 배치
        y = my + map_h + int(ch * 0.03)
        self.coords_title.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.08))
        y += int(ch * 0.07)
        self.lat_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))
        y += int(ch * 0.05)
        self.lng_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))

        # 버튼: 카드 하단 중앙
        bw = int(cw * self.BTN_W_PCT)
        bh = int(self.BTN_H_PX)
        bx = cx + (cw - bw) // 2
        by = cy + ch - bh - int(ch * self.BTN_BOTTOM_PCT)

        self.btn_send.setFixedSize(bw, bh)
        self.btn_send.setIconSize(QSize(int(bw * self.BTN_ICON_SCALE),
                                        int(bh * self.BTN_ICON_SCALE)))
        self.btn_send.move(bx, by)


        # 상단 아이콘(배경 상단바 상대 위치)
        for btn, x, y0, w, h in (
            (self.btn_home, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(QRect(0,0,self.BASE_W,self.BASE_H), x, y0, w=w, h=h, out_rect=full)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

        # Z-order
        self.bg.lower()
        self.web.raise_()
        self.btn_send.raise_()
        for b in (self.btn_home, self.btn_nav, self.btn_sos):
            b.raise_()


    def _card_rect(self, out_rect: QRect) -> tuple[int,int,int,int]:
        """배경 속 카드가 차지하는 화면 비율(CARD_PCT)을 실제 좌표로 변환."""
        px, py, pw, ph = self.CARD_PCT
        cw = int(out_rect.width()  * pw)
        ch = int(out_rect.height() * ph)
        cx = int(out_rect.width()  * px)
        cy = int(out_rect.height() * py)
        return cx, cy, cw, ch

    def _apply_round_mask(self, widget: QWidget, radius: int = 16):
        r = widget.rect()
        path = QPainterPath()
        path.addRoundedRect(0, 0, r.width(), r.height(), radius, radius)
        widget.setMask(QRegion(path.toFillPolygon().toPolygon()))

    def _map_from_design(self, design_rect: QRect, x, y, w=None, h=None, *, right=None, bottom=None, out_rect: QRect):
        sx = out_rect.width()  / design_rect.width()
        sy = out_rect.height() / design_rect.height()
        X = int(round(x * sx)); Y = int(round(y * sy))
        if w is not None and h is not None:
            W = int(round(w * sx)); H = int(round(h * sy))
        else:
            W = out_rect.width()  - X - int(round((right or 0)  * sx))
            H = out_rect.height() - Y - int(round((bottom or 0) * sy))
        return QRect(X, Y, W, H)
