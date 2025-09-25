from PySide6.QtCore import QObject, Signal, QTimer
from PySide6.QtPositioning import QGeoPositionInfoSource, QGeoPositionInfo
import threading, requests

class LocationProvider(QObject):
    gotLocation = Signal(float, float, str)   # lat, lng, label
    failed = Signal(str)

    def __init__(self, parent=None, timeout_ms=3500):
        super().__init__(parent)
        self._src = QGeoPositionInfoSource.createDefaultSource(self)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._on_timeout)
        self._timeout_ms = timeout_ms

        if self._src:
            self._src.setPreferredPositioningMethods(
                QGeoPositionInfoSource.AllPositioningMethods
            )
            self._src.positionUpdated.connect(self._on_pos)
            self._src.errorOccurred.connect(lambda e: None)

    def start(self):
        if self._src:
            self._timer.start(self._timeout_ms)
            self._src.startUpdates()
        else:
            self._fallback_ip()

    def _on_pos(self, info: QGeoPositionInfo):
        if not info.isValid():
            return
        self._timer.stop()
        if self._src:
            self._src.stopUpdates()
        c = info.coordinate()
        self.gotLocation.emit(c.latitude(), c.longitude(), "현위치")

    def _on_timeout(self):
        if self._src:
            self._src.stopUpdates()
        self._fallback_ip()

    def _fallback_ip(self):

        def work():
            try:
                # 후보 1: ipinfo
                r = requests.get("https://ipinfo.io/json", timeout=3)
                if r.ok:
                    data = r.json()
                    if "loc" in data:
                        lat, lng = map(float, data["loc"].split(","))
                        label = data.get("city") or "현재위치(대략)"
                        self.gotLocation.emit(lat, lng, label)
                        return
                # 후보 2: ip-api
                r = requests.get("http://ip-api.com/json", timeout=3)
                if r.ok:
                    data = r.json()
                    if data.get("status") == "success":
                        lat, lng = float(data["lat"]), float(data["lon"])
                        label = data.get("city") or "현재위치(대략)"
                        self.gotLocation.emit(lat, lng, label)
                        return
                # 실패 → 서울역
                self.failed.emit("IP geolocation failed")
            except Exception as e:
                self.failed.emit(str(e))
        threading.Thread(target=work, daemon=True).start()
