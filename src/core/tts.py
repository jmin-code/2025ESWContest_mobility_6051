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
    """
    gTTS -> 메모리 버퍼 -> mpg123(stdin) 재생.
    mpg123가 없으면 macOS 'afplay' 또는 ffplay로 폴백.
    """
    if not text or not text.strip():
        print("[tts] empty text; skip")
        return

    tts_text = ". " + text  # 앞머리 절두 방지용
    tts = gTTS(text=tts_text, lang=lang)

    mpg = _find_mpg123()
    if mpg:
        print(f"[tts] using mpg123 at: {mpg}")
        buf = io.BytesIO()
        tts.write_to_fp(buf); buf.seek(0)
        try:
            proc = subprocess.Popen(
                [mpg, "-q", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,   # <- 오류 잡아서 로깅
            )
            assert proc.stdin is not None
            proc.stdin.write(buf.read())
            proc.stdin.close()
            _, err = proc.communicate()
            if proc.returncode != 0:
                print(f"[tts] mpg123 exited {proc.returncode}, stderr: {err.decode('utf-8', 'ignore')}")
        except FileNotFoundError:
            mpg = None  # 아래 폴백으로 이동
        except Exception as e:
            print(f"[tts] mpg123 failed: {e}")
            mpg = None

    if not mpg:
        # 폴백: 임시 mp3 파일 만들어서 재생
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_name = tmp.name
        try:
            tts.write_to_fp(tmp)
            tmp.flush(); tmp.close()

            # macOS 기본 재생기
            if _which("afplay"):
                print("[tts] fallback -> afplay")
                subprocess.run(["afplay", tmp_name], check=False)
            # ffplay가 있으면 GUI 없이
            elif _which("ffplay"):
                print("[tts] fallback -> ffplay")
                subprocess.run(["ffplay", "-autoexit", "-nodisp", tmp_name], check=False)
            else:
                raise RuntimeError("No audio player found (need mpg123 or afplay/ffplay).")
        finally:
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
