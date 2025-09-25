from collections import deque
from enum import Enum, auto

# --- Constants ---
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONGSUNG_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

HANGUL_LABELS = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
    8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ', 14: 'ㅏ', 15: 'ㅑ',
    16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ', 21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ',
    24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ', 28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ', 31: 'end',
    32: 'ㄲ', 33: 'ㄸ', 34: 'ㅃ', 35: 'ㅆ', 36: 'ㅉ', 47: ' ', 48: 'backspace'
}

# 집합(Set)으로 자/모음 확인 성능 향상
CHOSUNG_SET = set(CHOSUNG_LIST)
JUNGSUNG_SET = set(JUNGSUNG_LIST)
JONGSUNG_SET = set(JONGSUNG_LIST)

# 복합 자/모음 정의
COMPOUND_JUNGSUNG = {('ㅗ', 'ㅏ'): 'ㅘ', ('ㅗ', 'ㅐ'): 'ㅙ', ('ㅗ', 'ㅣ'): 'ㅚ',
                     ('ㅜ', 'ㅓ'): 'ㅝ', ('ㅜ', 'ㅔ'): 'ㅞ', ('ㅜ', 'ㅣ'): 'ㅟ',
                     ('ㅡ', 'ㅣ'): 'ㅢ'}
COMPOUND_JONGSUNG = {('ㄱ', 'ㅅ'): 'ㄳ', ('ㄴ', 'ㅈ'): 'ㄵ', ('ㄴ', 'ㅎ'): 'ㄶ',
                     ('ㄹ', 'ㄱ'): 'ㄺ', ('ㄹ', 'ㅁ'): 'ㄻ', ('ㄹ', 'ㅂ'): 'ㄼ',
                     ('ㄹ', 'ㅅ'): 'ㄽ', ('ㄹ', 'ㅌ'): 'ㄾ', ('ㄹ', 'ㅍ'): 'ㄿ',
                     ('ㄹ', 'ㅎ'): 'ㅀ', ('ㅂ', 'ㅅ'): 'ㅄ'}

class State(Enum):
    INITIAL = auto()
    AWAITING_VOWEL = auto()
    AWAITING_CONSONANT = auto()
    PENDING_JONG = auto()

class HangulComposer:
    def __init__(self):
        self.state = State.INITIAL
        self.buffer = [None, None, None]
        self.result_queue = deque()
        self.current_char_display = ""
        self.CONSONANT_REPEAT_THRESHOLD = 110

    def _combine_hangul(self, cho, ju, jo=None):
        try:
            cho_idx = CHOSUNG_LIST.index(cho)
            jung_idx = JUNGSUNG_LIST.index(ju)
            jong_idx = JONGSUNG_LIST.index(jo) if jo else 0
            return chr(0xAC00 + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx)
        except (ValueError, IndexError, TypeError):
            return cho if cho else None

    def _update_display(self):
        cho, jung, jong = self.buffer
        self.current_char_display = self._combine_hangul(cho, jung, jong) or (cho if cho else "")

    def flush_all_pending(self):
        if self.buffer[0]:
            self._update_display()
            if self.current_char_display:
                self.result_queue.append(self.current_char_display)
        self.buffer = [None, None, None]
        self.state = State.INITIAL
        self._update_display()

    def process_input(self, label: str, frame_count: int = 0):
        if label in CHOSUNG_SET:
            self._handle_consonant(label, frame_count)
        elif label in JUNGSUNG_SET:
            self._handle_vowel(label)
        elif label in [' ', 'backspace']:
            self._handle_special(label)
        self._update_display()

    def _handle_consonant(self, label, frame_count):
        if self.state == State.PENDING_JONG:
            if label == self.buffer[2] and frame_count >= self.CONSONANT_REPEAT_THRESHOLD:
                self.flush_all_pending()
                self.buffer = [label, None, None]
                self.state = State.AWAITING_VOWEL
            else:
                self.flush_all_pending()
                self.buffer = [label, None, None]
                self.state = State.AWAITING_VOWEL
        elif self.state == State.AWAITING_CONSONANT:
            if not self.buffer[2] and label in JONGSUNG_SET:
                self.buffer[2] = label
                self.state = State.PENDING_JONG
            else:
                self.flush_all_pending()
                self.buffer = [label, None, None]
                self.state = State.AWAITING_VOWEL
        else:
            self.flush_all_pending()
            self.buffer = [label, None, None]
            self.state = State.AWAITING_VOWEL

    def _handle_vowel(self, label):
        if self.state == State.PENDING_JONG:
            cho, jung, jong = self.buffer
            char_without_jong = self._combine_hangul(cho, jung)
            if char_without_jong:
                self.result_queue.append(char_without_jong)
            self.buffer = [jong, label, None]
            self.state = State.AWAITING_CONSONANT
        elif self.state == State.AWAITING_VOWEL:
            self.buffer[1] = label
            self.state = State.AWAITING_CONSONANT
        elif self.state == State.AWAITING_CONSONANT:
            if self.buffer[1] and (self.buffer[1], label) in COMPOUND_JUNGSUNG:
                self.buffer[1] = COMPOUND_JUNGSUNG[(self.buffer[1], label)]
            else:
                self.flush_all_pending()
                self.result_queue.append(label)
                self.state = State.INITIAL

    def _handle_special(self, label):
        if label == ' ':
            self.flush_all_pending()
            self.result_queue.append(' ')
        elif label == 'backspace':
            # *** 여기가 수정된 부분입니다 ***
            if self.state == State.PENDING_JONG:
                # 종성 입력 대기 상태: 종성만 삭제
                self.buffer[2] = None
                self.state = State.AWAITING_CONSONANT
            elif self.state == State.AWAITING_CONSONANT:
                # 중성 입력 완료 상태: 중성 삭제
                self.buffer[1] = None
                self.state = State.AWAITING_VOWEL
            elif self.state == State.AWAITING_VOWEL:
                # 초성 입력 완료 상태: 초성 삭제
                self.buffer[0] = None
                self.state = State.INITIAL
            elif self.state == State.INITIAL and self.result_queue:
                # 완성된 글자만 있는 경우: 마지막 글자 전체 삭제
                self.result_queue.pop()

    def get_current_char(self) -> str:
        return self.current_char_display