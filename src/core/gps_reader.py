# src/core/gps_reader.py

import sys
import serial
import time
from PySide6.QtCore import QObject, Signal, QThread

class GPSReader(QObject):
    """
    별도의 스레드에서 GPS 모듈의 시리얼 데이터를 읽어 파싱하고,
    유효한 좌표가 수신되면 시그널을 발생시키는 클래스.
    """
    # 새 좌표를 받았을 때 발생할 시그널 정의 (위도, 경도)
    new_location = Signal(float, float)
    
    def __init__(self, port='/dev/ttyACM0', baudrate=9600, parent=None):
        super().__init__(parent)
        self.port = port
        self.baudrate = baudrate
        self.running = False
        self.serial_port = None
        print(f"✅ GPSReader initialized for port {self.port}")

    def run(self):
        """이 함수가 백그라운드 스레드에서 실행됩니다."""
        self.running = True
        while self.running:
            try:
                self.serial_port = serial.Serial(self.port, self.baudrate, timeout=1)
                print(f"✅ GPS Port {self.port} opened successfully.")
                
                while self.running:
                    line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('$GPGGA'):
                        parts = line.split(',')
                        if len(parts) > 6 and parts[2] and parts[4]:
                            lat = self.convert_to_decimal(parts[2], parts[3])
                            lng = self.convert_to_decimal(parts[4], parts[5])
                            if lat and lng:
                                # print(f"🛰️ GPS Location Found: Lat={lat}, Lng={lng}")
                                self.new_location.emit(lat, lng)
                    time.sleep(0.1) # CPU 사용량을 줄이기 위한 짧은 대기
            
            except serial.SerialException as e:
                print(f"❌ GPS Port Error: {e}. Retrying in 5 seconds...")
                time.sleep(5)
            except Exception as e:
                print(f"❌ An unexpected error occurred in GPSReader: {e}")
                time.sleep(5)

    def stop(self):
        self.running = False
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("✅ GPS Port closed.")

    def convert_to_decimal(self, nmea_val, direction):
        """NMEA 형식(DDMM.MMMM)의 좌표를 십진수(DD.DDDDDD)로 변환합니다."""
        try:
            raw_val = float(nmea_val)
            degrees = int(raw_val / 100)
            minutes = raw_val % 100
            decimal = degrees + (minutes / 60)
            if direction in ['S', 'W']:
                decimal *= -1
            return decimal
        except (ValueError, IndexError):
            return None