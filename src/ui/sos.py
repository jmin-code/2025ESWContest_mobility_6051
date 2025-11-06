# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import socket
import threading

from PySide6.QtCore import Qt, QRect, QSize, QUrl, QTimer, Slot
from PySide6.QtGui import QPixmap, QIcon, QRegion, QPainterPath, QFont, QFontDatabase
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage

FONT_PATH = Path("/home/autonav/2025ESWContest_mobility_6051/src/ui/assets/fonts/SourceSans3-SemiBold.ttf")

# ==== SOS 전송 설정 ====
SERVER_IP = "10.0.0.22"
SERVER_PORT = 12345

# GPS 없음 기본 위치(인하대학교 예시)
DEFAULT_LATITUDE = 37.576738
DEFAULT_LONGITUDE = 126.897859


class SOSPage(QWidget):
    """
    NavigationPage와 동일한 800x480 디자인 좌표계를 사용.
    배경은 fit 사각형에 비율 유지로 넣고,
    상단바 버튼은 fit 기준 _map_from_design()으로 배치.
    카드 내부 요소(지도/텍스트/버튼)는 퍼센트 기반으로 풀창에 비례 배치.
    """
    BASE_W, BASE_H = 800, 480

    # 카드(배경 이미지 내 카드 영역) 위치/크기 비율: (x%, y%, w%, h%)
    CARD_PCT = (0.22, 0.23, 0.56, 0.70)

    # 카드 내부 구성 요소 비율
    MAP_W_PCT  = 0.67
    MAP_H_PCT  = 0.60
    MAP_TOP_PCT = 0.05
    BTN_W_PCT  = 0.3
    BTN_H_PX   = 45
    BTN_BOTTOM_PCT = 0.045
    BTN_ICON_SCALE = 1.0

    def __init__(self, assets_dir: Path, on_home=None, on_voice=None, on_nav=None, on_sos=None, on_send=None, fonts=None, sign_engine=None):
        super().__init__()
        self.assets = Path(assets_dir)
        self.on_home  = on_home  or (lambda: None)
        self.on_voice = on_voice or (lambda: None)
        self.on_nav   = on_nav   or (lambda: None)
        self.on_sos   = on_sos   or (lambda: None)
        self.original_on_send = on_send or (lambda: None)

        self.fonts = fonts or {}
        self.sign_engine = sign_engine

        # --- 배경 ---
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = self._load_pix(self.assets / "bg" / "sos_bg.png")

        # --- 폰트 로드 ---
        self._font_family = None
        try:
            fid = QFontDatabase.addApplicationFont(str(FONT_PATH))
            fams = QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
            if fams:
                self._font_family = fams[0]  # 예: "Source Sans 3"
                print(f"[SOS] Font loaded: {self._font_family}")
            else:
                print("[SOS] Font families not found, using fallback.")
        except Exception as e:
            print(f"[SOS] Font load error: {e}")


        # --- 상단 아이콘 (navigation.py와 동일한 제작 방식) ---
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = self._load_pix(self.assets / "icons" / fname)
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            if cb: b.clicked.connect(cb)
            return b
        

        self.btn_home  = mk_icon("home.png",        self.on_home)
        self.btn_voice = mk_icon("voice_over.png",  self.on_voice)
        self.btn_nav   = mk_icon("nav.png",         self.on_nav)
        self.btn_sos   = mk_icon("sos_b.png",       lambda: None)  # 현재 페이지 active

        # --- 카드 내부 요소 ---
        self.web = QWebEngineView(self)
        self.web.setPage(QWebEnginePage(self.web))
        self.web.loadFinished.connect(self._on_map_loaded)

        self.coords_title = QLabel("Current Location Coordinates:", self)
        self.coords_title.setAlignment(Qt.AlignCenter)
        self.coords_title.setStyleSheet("color:#CFCFCF;")

        self.lat_label = QLabel("Latitude: —", self)
        self.lat_label.setAlignment(Qt.AlignCenter)
        self.lat_label.setStyleSheet("color:#DADADA;")

        self.lng_label = QLabel("Longitude: —", self)
        self.lng_label.setAlignment(Qt.AlignCenter)
        self.lng_label.setStyleSheet("color:#DADADA;")

        # 폰트(페이지 공통 폰트가 있으면 사용)
        base_family = self._font_family or self.fonts.get("korean", "Arial")
        self.coords_title.setFont(QFont(base_family, 13))
        self.lat_label.setFont(   QFont(base_family, 12))
        self.lng_label.setFont(   QFont(base_family, 12))


        # SOS 버튼
        self.btn_send = QPushButton(self)
        self.btn_send.setCursor(Qt.PointingHandCursor)
        self.btn_send.setStyleSheet("QPushButton{border:none;background:transparent}")
        pm_send = self._load_pix(self.assets / "icons" / "send_sos.png")
        if not pm_send.isNull():
            self.btn_send.setIcon(QIcon(pm_send))
        else:
            self.btn_send.setText("Send SOS")
            self.btn_send.setStyleSheet("QPushButton { color: white; background:#E53935; border-radius: 28px; padding:14px 28px; }")
        self.btn_send.clicked.connect(self._send_current_location)

        # 상태
        self._latlng: Optional[Tuple[float, float]] = None
        self._map_ready = False
        self._initial_loaded = False

        # 레이아웃 1회 적용 및 초기 로드
        self._relayout()
        QTimer.singleShot(1000, self._set_default_location_if_needed)
        QTimer.singleShot(0, self._load_initial_map)


    # ========================
    # 공통 유틸 (navigation.py와 동일 패턴)
    # ========================
    def _load_pix(self, path: Path) -> QPixmap:
        pm = QPixmap(str(path))
        if not pm.isNull() and (pm.width() >= self.BASE_W*2 or pm.height() >= self.BASE_H*2):
            pm.setDevicePixelRatio(2.0)
        return pm

    def _fit_rect_for_pixmap(self, pm: QPixmap, box: QRect) -> QRect:
        if pm.isNull(): return box
        dpr = pm.devicePixelRatio() or 1.0
        w0, h0 = int(pm.width()/dpr), int(pm.height()/dpr)
        bw, bh = box.width(), box.height()
        if not all((w0, h0, bw, bh)): return box
        s = min(bw/w0, bh/h0)
        w, h = int(w0*s), int(h0*s)
        return QRect(box.x()+(bw-w)//2, box.y()+(bh-h)//2, w, h)

    def _map_from_design(self, fit: QRect, x, y, w=None, h=None, *, right=None, bottom=None) -> QRect:
        """navigation.py와 동일한 시그니처"""
        sx, sy = fit.width()/self.BASE_W, fit.height()/self.BASE_H
        X, Y = fit.x()+int(round(x*sx)), fit.y()+int(round(y*sy))
        if w is not None and h is not None:
            W, H = int(round(w*sx)), int(round(h*sy))
        else:
            W = fit.width()  - (X - fit.x()) - int(round((right or 0)*sx))
            H = fit.height() - (Y - fit.y()) - int(round((bottom or 0)*sy))
        return QRect(X, Y, W, H)

    # ========================
    # 기능 로직
    # ========================
    @Slot(float, float)
    def set_location(self, lat: float, lng: float):
        if self._latlng == (lat, lng):
            return
        self._latlng = (lat, lng)
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lng >= 0 else "W"
        self.lat_label.setText(f"Latitude: {abs(lat):.4f}° {ns}")
        self.lng_label.setText(f"Longitude: {abs(lng):.4f}° {ew}")
        if self._map_ready:
            self._update_map_location(lat, lng)

    def _set_default_location_if_needed(self):
        if self._latlng is None:
            print("[SOS] GPS 신호 없음. 기본 위치(인하대학교)로 설정합니다.")
            self.set_location(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)

    def _send_current_location(self):
        if self._latlng is None:
            print("[SOS] 위치 정보가 없어 전송할 수 없습니다.")
            return
        # 안전 가드: IP 잘못 설정시 전송만 막고 화면전환은 하지 않음
        if not SERVER_IP or SERVER_IP.strip() == "10":
            print("[SOS] 경고: SERVER_IP를 실제 수신 컴퓨터의 IP로 변경하세요.")
            return

        thread = threading.Thread(target=self._send_location_thread, args=self._latlng)
        thread.daemon = True
        thread.start()
        self.original_on_send()

    def _send_location_thread(self, lat: float, lng: float):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((SERVER_IP, SERVER_PORT))
                msg = f"{lat},{lng}"
                s.sendall(msg.encode("utf-8"))
                print(f"[SOS] 위치 전송 성공: {msg} to {SERVER_IP}:{SERVER_PORT}")
        except ConnectionRefusedError:
            print(f"[SOS] 오류: 서버에 연결할 수 없습니다. ({SERVER_IP}:{SERVER_PORT})")
        except socket.timeout:
            print(f"[SOS] 오류: 서버 연결 시간 초과. ({SERVER_IP}:{SERVER_PORT})")
        except Exception as e:
            print(f"[SOS] 위치 전송 중 오류 발생: {e}")

    def _load_initial_map(self):
        if self._initial_loaded: return
        self._initial_loaded = True
        # search.html에 updateCurrentLocation()이 있어 현위치 표시 가능
        self.web.load(QUrl("http://localhost:5050/search.html"))

    @Slot(bool)
    def _on_map_loaded(self, ok: bool):
        if not ok:
            print("[SOS] 맵 로드 실패!")
            return
        self._map_ready = True
        lat, lng = self._latlng if self._latlng else (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self._update_map_location(lat, lng)

    def _update_map_location(self, lat: float, lng: float):
        js = f"updateCurrentLocation({lat}, {lng});"
        self.web.page().runJavaScript(js)

    # ========================
    # 레이아웃
    # ========================
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def _relayout(self):
        full = self.rect()
        # 배경: navigation.py처럼 fit 계싼 후 넣기
        fit = self._fit_rect_for_pixmap(self.pm_bg, full)
        self.bg.setGeometry(fit)
        if not self.pm_bg.isNull():
            dpr = self.pm_bg.devicePixelRatio() or 1.0
            img = self.pm_bg.toImage().scaled(int(fit.width()*dpr), int(fit.height()*dpr),
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pm2 = QPixmap.fromImage(img); pm2.setDevicePixelRatio(dpr)
            self.bg.setPixmap(pm2)
        else:
            self.bg.setStyleSheet("background:#111")

        # 카드 영역(퍼센트 기반, 전체 화면 기준)
        cx, cy, cw, ch = self._card_rect(full)

        self._apply_font_sizes(ch)

        # 지도 영역
        pad = int(cw * 0.06)
        y_top  = int(ch * self.MAP_TOP_PCT)
        map_w  = int(cw * self.MAP_W_PCT)
        map_h  = int(ch * self.MAP_H_PCT)
        mx = cx + (cw - map_w) // 2
        my = cy + y_top
        self.web.setGeometry(mx, my, map_w, map_h)
        self._apply_round_mask(self.web, radius=22)

        # 좌표 라벨
        y = my + map_h + int(ch * 0.03)
        self.coords_title.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.08))
        y += int(ch * 0.07)
        self.lat_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))
        y += int(ch * 0.05)
        self.lng_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))

        # SOS 버튼
        bw = int(cw * self.BTN_W_PCT)
        bh = int(self.BTN_H_PX)
        bx = cx + (cw - bw) // 2
        by = cy + ch - bh - int(ch * self.BTN_BOTTOM_PCT)
        self.btn_send.setFixedSize(bw, bh)
        self.btn_send.setIconSize(QSize(int(bw * self.BTN_ICON_SCALE), int(bh * self.BTN_ICON_SCALE)))
        self.btn_send.move(bx, by)

        # 상단 아이콘(네비게이션과 동일: fit 기준 매핑)
        for btn, x, y0, w, h in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(fit, x, y0, w=w, h=h)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())

        # Z-순서
        self.bg.lower()
        self.web.raise_()
        self.btn_send.raise_()
        for b in (self.btn_home, self.btn_nav, self.btn_sos, self.btn_voice):
            b.raise_()
        
    def _apply_font_sizes(self, card_h: int):
        title_px = max(10, int(card_h * 0.048))
        value_px = max(9, int(card_h * 0.045))

        # 로드된 패밀리 > 전달받은 fonts 사전 > 시스템 폴백 순서
        fam = self._font_family or self.fonts.get("korean", "Arial")
        fam_css = f"font-family: '{fam}';"

        self.coords_title.setStyleSheet(
            f"{fam_css} color:#CFCFCF; font-weight:700; font-size:{title_px}px;"
        )
        self.lat_label.setStyleSheet(
            f"{fam_css} color:#DADADA; font-weight:600; font-size:{value_px}px;"
        )
        self.lng_label.setStyleSheet(
            f"{fam_css} color:#DADADA; font-weight:600; font-size:{value_px}px;"
        )




    def _card_rect(self, out_rect: QRect) -> tuple[int,int,int,int]:
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
