import socket

HOST = '0.0.0.0'  # 모든 IP에서 오는 연결을 허용
PORT = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"서버가 시작되었습니다. {HOST}:{PORT}에서 대기 중...")
    while True:
        conn, addr = s.accept()
        with conn:
            print(f"{addr}에서 연결됨")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                # 수신된 데이터(위도,경도)를 출력
                location_str = data.decode('utf-8')
                print(f"🚑 긴급 구조 요청 위치 ‼️ : {location_str}")