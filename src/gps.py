# gps.py

from flask import Flask, Response, render_template, jsonify
import serial
import time
import json
import threading
import os

# --- Flask 웹 서버 설정 ---
app = Flask(__name__)

# --- 실시간 GPS 데이터를 저장할 변수 ---
gps_data_lock = threading.Lock()
latest_gps_data = {
    "lat": 37.6005,  # 초기값: 인하대학교
    "lon": 126.72,
    "spd": 0
}

# --- 백그라운드에서 계속 GPS 데이터를 읽을 함수 ---
def read_from_gps(port="/dev/ttyACM0"):
    global latest_gps_data
    while True:
        try:
            print(f"GPS 장치에 연결 시도 중... ({port})")
            ser = serial.Serial(port, 115200, timeout=1)
            print("✅ GPS 수신 대기 중...", ser.port)
            
            while True:
                raw = ser.readline()
                if not raw: continue
                raw = raw.replace(b'\x00', b'').strip()
                if not raw: continue

                try:
                    text = raw.decode('utf-8')
                    parts = [p.strip() for p in text.split(',')]
                    if len(parts) >= 3:
                        lat, lon, spd = float(parts[0]), float(parts[1]), float(parts[2])
                        
                        with gps_data_lock:
                            latest_gps_data["lat"] = lat
                            latest_gps_data["lon"] = lon
                            latest_gps_data["spd"] = spd
                        
                        print(f"🛰️  수신: lat={lat:.6f}, lon={lon:.6f}, spd={spd:.1f} km/h", flush=True) # 디버깅용

                except (UnicodeDecodeError, ValueError, IndexError):
                    pass
        
        except serial.SerialException:
            print(f"❌ 오류: 시리얼 포트 '{port}'를 열 수 없습니다. 5초 후 재시도합니다.")
            time.sleep(5)
        except Exception as e:
            print(f"GPS 스레드에서 예외 발생: {e}")
            time.sleep(5)


# --- 파이썬 클라이언트용 API 경로 ---
@app.route('/api/gps')
def api_gps():
    with gps_data_lock:
        return jsonify(latest_gps_data.copy())

# --- 웹 브라우저용 실시간 스트림 경로 ---
@app.route('/gps-stream')
def gps_stream():
    def event_stream():
        while True:
            with gps_data_lock:
                yield f"data: {json.dumps(latest_gps_data)}\n\n"
            time.sleep(1)
    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    # GPS 데이터 읽기를 백그라운드 스레드에서 시작
    gps_thread = threading.Thread(target=read_from_gps, args=(os.getenv("GPS_PORT", "/dev/ttyACM0"),), daemon=True)
    gps_thread.start()
    
    # Flask 웹 서버 시작
    print("✅ GPS 데이터 서버를 시작합니다 (http://127.0.0.1:6051)")
    app.run(host='0.0.0.0', port=6051, threaded=True)