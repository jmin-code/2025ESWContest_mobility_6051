# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
import socket
import threading

from PySide6.QtCore import Qt, QRect, QSize, QUrl, QTimer, Slot
from PySide6.QtGui import QPixmap, QIcon, QRegion, QPainterPath
from PySide6.QtWidgets import QWidget, QLabel, QPushButton
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEnginePage


# <<< 중요!! >>>
# 아래 IP 주소를 위치 정보를 수신할 컴퓨터의 실제 IP 주소로 반드시 변경해야 함
# 예: SERVER_IP = "192.168.0.10"
SERVER_IP = "10.0.0.23"
SERVER_PORT = 12345

# GPS 신호가 없을 때 사용할 기본 위치 (예: 인하대학교)
DEFAULT_LATITUDE = 37.449120
DEFAULT_LONGITUDE = 126.655856


class SOSPage(QWidget):
    """
    배경(sos_bg.png)에 카드가 이미 포함되어 있으므로 별도 카드 위젯을 만들지 않는다.
    배경 이미지의 카드 영역을 비율로 계산해 그 안에 지도/라벨/버튼만 배치.
    """
    BASE_W, BASE_H = 800, 480
    CARD_PCT = (0.22, 0.23, 0.56, 0.70)
    MAP_W_PCT = 0.74
    MAP_H_PCT = 0.60
    MAP_TOP_PCT = 0.05
    BTN_W_PCT = 0.28
    BTN_H_PX = 40
    BTN_BOTTOM_PCT = 0.005
    BTN_ICON_SCALE = 0.9

    def __init__(self, assets_dir: Path, on_home=None, on_voice=None, on_nav=None, on_send=None):
        super().__init__()
        self.assets = Path(assets_dir)
        self.on_home = on_home or (lambda: None)
        self.on_voice = on_voice or (lambda: None)
        self.on_nav = on_nav or (lambda: None)
        self.original_on_send = on_send or (lambda: None)

        # --- 배경 ---
        self.bg = QLabel(self); self.bg.setAlignment(Qt.AlignCenter)
        self.pm_bg = QPixmap(str(self.assets / "bg" / "sos_bg.png"))

        # --- 상단 아이콘 ---
        def mk_icon(fname, cb):
            b = QPushButton(self)
            pm = QPixmap(str(self.assets / "icons" / fname))
            if not pm.isNull(): b.setIcon(QIcon(pm))
            b.setStyleSheet("border:none;background:transparent")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(cb)
            return b
        self.btn_home = mk_icon("home.png", self.on_home)
        self.btn_voice = mk_icon("voice_over.png", self.on_voice)
        self.btn_nav = mk_icon("nav.png", self.on_nav)
        self.btn_sos = mk_icon("sos_b.png", lambda: None)

        # --- 카드 내부 요소 ---
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
        
        # SOS 보내기 버튼을 누르면 _send_current_location 함수 실행
        self.btn_send.clicked.connect(self._send_current_location)

        # 상태 변수
        self._latlng: Optional[Tuple[float, float]] = None
        self._map_ready = False
        self._initial_loaded = False

        self._relayout()
        # GPS 신호가 없을 경우를 대비해 타이머로 기본 위치 설정
        QTimer.singleShot(1000, self._set_default_location_if_needed)
        QTimer.singleShot(0, self._load_initial_map)

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

        if self._map_ready:
            self._update_map_location(lat, lng)

    def _set_default_location_if_needed(self):
        """1초 후에도 GPS 위치가 없으면 기본 위치(인하대)로 설정"""
        if self._latlng is None:
            print("[SOS] GPS 신호 없음. 기본 위치(인하대학교)로 설정합니다.")
            self.set_location(DEFAULT_LATITUDE, DEFAULT_LONGITUDE)

    def _send_current_location(self):
        """현재 위치를 다른 컴퓨터로 전송하는 기능"""
        if self._latlng is None:
            print("[SOS] 위치 정보가 없어 전송할 수 없습니다.")
            return

        # 사용자가 IP 주소를 변경했는지 확인
        if SERVER_IP == "10":
            print("[SOS] 경고: 코드 상단의 SERVER_IP를 실제 수신 컴퓨터의 IP로 변경해야 합니다.")
            # IP가 기본값이면 전송 후 화면 전환(original_on_send)을 막아 사용자가 문제를 인지하게 함
            return

        # UI가 멈추지 않도록 별도 스레드에서 소켓 통신 실행
        thread = threading.Thread(target=self._send_location_thread, args=self._latlng)
        thread.daemon = True
        thread.start()
        
        # 위치 전송 후 원래 지정된 콜백 함수(화면 전환 등) 실행
        self.original_on_send()

    def _send_location_thread(self, lat: float, lng: float):
        """소켓을 이용해 위치 정보를 서버로 전송하는 스레드 함수"""
        try:
            # TCP 소켓 생성
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5) # 5초 연결 타임아웃
                s.connect((SERVER_IP, SERVER_PORT))
                message = f"{lat},{lng}"
                s.sendall(message.encode('utf-8'))
                print(f"[SOS] 위치 전송 성공: {message} to {SERVER_IP}:{SERVER_PORT}")
        except ConnectionRefusedError:
            print(f"[SOS] 오류: 서버에 연결할 수 없습니다. ({SERVER_IP}:{SERVER_PORT})")
        except socket.timeout:
            print(f"[SOS] 오류: 서버 연결 시간 초과. ({SERVER_IP}:{SERVER_PORT})")
        except Exception as e:
            print(f"[SOS] 위치 전송 중 오류 발생: {e}")

    def _font(self, size: int, bold: bool=False):
        f = self.font()
        f.setPointSize(size)
        f.setBold(bold)
        return f

    def _load_initial_map(self):
        if self._initial_loaded: return
        self._initial_loaded = True
        self.web.load(QUrl("http://localhost:5050/search.html"))

    @Slot(bool)
    def _on_map_loaded(self, ok: bool):
        if not ok:
            print("[SOS] 맵 로드 실패!")
            return
        self._map_ready = True

        QTimer.singleShot(100, self._update_map_to_current_location)

    def _update_map_to_current_location(self):
        if not self._map_ready:
            return
        lat, lng = self._latlng if self._latlng else (DEFAULT_LATITUDE, DEFAULT_LONGITUDE)
        self._update_map_location(lat, lng)


    def _update_map_location(self, lat: float, lng: float):
        js_code = f"window.updateCurrentLocation({lat}, {lng});"
        self.web.page().runJavaScript(js_code)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._relayout()

    def _relayout(self):
        full = self.rect()
        self.bg.setGeometry(full)
        if not self.pm_bg.isNull():
            self.bg.setPixmap(self.pm_bg.scaled(full.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else: self.bg.setStyleSheet("background:#111")
        cx, cy, cw, ch = self._card_rect(full)
        pad = int(cw * 0.06)
        y = int(ch * self.MAP_TOP_PCT)
        map_w = int(cw * self.MAP_W_PCT)
        map_h = int(ch * self.MAP_H_PCT)
        mx = cx + (cw - map_w) // 2
        my = cy + y
        self.web.setGeometry(mx, my, map_w, map_h)
        self._apply_round_mask(self.web, radius=22)
        y = my + map_h + int(ch * 0.03)
        self.coords_title.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.08))
        y += int(ch * 0.07)
        self.lat_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))
        y += int(ch * 0.05)
        self.lng_label.setGeometry(cx + pad, y, cw - 2*pad, int(ch * 0.07))
        bw = int(cw * self.BTN_W_PCT)
        bh = int(self.BTN_H_PX)
        bx = cx + (cw - bw) // 2
        by = cy + ch - bh - int(ch * self.BTN_BOTTOM_PCT)
        self.btn_send.setFixedSize(bw, bh)
        self.btn_send.setIconSize(QSize(int(bw * self.BTN_ICON_SCALE), int(bh * self.BTN_ICON_SCALE)))
        self.btn_send.move(bx, by)
        for btn, x, y0, w, h in (
            (self.btn_home, 603, 20, 24, 24),
            (self.btn_voice, 653, 20, 24, 24),
            (self.btn_nav,  703, 20, 22, 22),
            (self.btn_sos,  753, 20, 22, 22),
        ):
            r = self._map_from_design(QRect(0,0,self.BASE_W,self.BASE_H), x, y0, w=w, h=h, out_rect=full)
            btn.setIconSize(QSize(r.width(), r.height()))
            btn.setFixedSize(r.width(), r.height())
            btn.move(r.x(), r.y())
        self.bg.lower()
        self.web.raise_()
        self.btn_send.raise_()
        for b in (self.btn_home, self.btn_nav, self.btn_sos, self.btn_voice):
            b.raise_()

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