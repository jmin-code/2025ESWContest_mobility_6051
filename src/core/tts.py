# -*- coding: utf-8 -*-
import io, os, shutil, tempfile, subprocess
from typing import Optional
from gtts import gTTS

def _which(prog: str) -> Optional[str]:
    p = shutil.which(prog)
    return p if p else None

def _find_mpg123() -> Optional[str]:
    # macOS 홈브류 경로까지 직접 탐색
    candidates = [
        "mpg123",
        "/opt/homebrew/bin/mpg123",
        "/usr/local/bin/mpg123",
        "/usr/bin/mpg123",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
        w = _which(c)
        if w:
            return w
    return None

def speak_stream(text: str, lang: str = "ko") -> None:
    if not text or not text.strip():
        print("[tts] empty text; skip")
        return

    tts_text = ". " + text
    tts = gTTS(text=tts_text, lang=lang)

    # 공통 버퍼 1회만 생성
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    data = buf.read()

    mpg = _find_mpg123()
    if mpg:
        print(f"[tts] using mpg123 at: {mpg}")
        try:
            proc = subprocess.Popen(
                [mpg, "-q", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None
            proc.stdin.write(data)
            proc.stdin.close()
            _, err = proc.communicate()
            if proc.returncode != 0:
                print(f"[tts] mpg123 exited {proc.returncode}, stderr: {err.decode('utf-8', 'ignore')}")
                mpg = None
        except Exception as e:
            print(f"[tts] mpg123 failed: {e}")
            mpg = None

    if not mpg:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_name = tmp.name
        try:
            tmp.write(data)  # 이미 만든 data 재사용
            tmp.flush(); tmp.close()

            if _which("afplay"):
                print("[tts] fallback -> afplay")
                subprocess.run(["afplay", tmp_name], check=False)
            elif _which("ffplay"):
                print("[tts] fallback -> ffplay")
                subprocess.run(["ffplay", "-autoexit", "-nodisp", tmp_name], check=False)
            else:
                raise RuntimeError("No audio player found (need mpg123 or afplay/ffplay).")
        finally:
            try: os.unlink(tmp_name)
            except: pass
