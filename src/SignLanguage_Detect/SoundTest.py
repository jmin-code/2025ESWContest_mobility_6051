import io
import subprocess
from gtts import gTTS

# --- 설정 ---
TEST_TEXT = " 안녕안녕안녕 이제 딜레이 없이 바로 소리가 나올 겁니다. 긴 문장을 테스트해도 처음부터 잘 들릴 거예요."

print("실시간 스트리밍 방식으로 음성 출력을 시작합니다...")

try: 
    # 1. gTTS 객체 생성
    tts = gTTS(text=TEST_TEXT, lang='ko')

    # 2. mpg123 프로세스를 파이프(pipe) 모드로 실행
    #    '-'는 파일 대신 표준 입력(stdin)으로부터 데이터를 읽으라는 의미입니다.
    #    ['mpg123', '-q', '-'] -> mpg123 프로그램을 조용히(-q) 실행하고, 표준 입력(-)을 기다림
    proc = subprocess.Popen(
        ['mpg123', '-v' ,'--buffer', '1024', '-'],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # 3. gTTS가 생성하는 음성 데이터를 파일이 아닌 메모리 버퍼(fp)에 쓰고,
    #    이것을 바로 mpg123 프로세스의 표준 입력(stdin)으로 보냅니다.
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0) # 버퍼의 커서를 처음으로 이동
    
    # 메모리에 있는 음성 데이터를 프로세스로 전달
    proc.stdin.write(fp.read())
    proc.stdin.close() # 데이터 전송이 끝났음을 알림
    
    proc.wait() # 재생이 끝날 때까지 기다림

    print("테스트 완료.")

except Exception as e:
    print(f"오류가 발생했습니다: {e}")