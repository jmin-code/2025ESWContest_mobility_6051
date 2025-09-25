#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- 1. 설정 (Configuration) ---
DATA_DIR = "../../hangul"

# HANGUL 사전을 sign_engine.py와 일치하도록 쌍자음 추가
HANGUL = {
    0: 'ㄱ', 1: 'ㄴ', 2: 'ㄷ', 3: 'ㄹ', 4: 'ㅁ', 5: 'ㅂ', 6: 'ㅅ', 7: 'ㅇ',
    8: 'ㅈ', 9: 'ㅊ', 10: 'ㅋ', 11: 'ㅌ', 12: 'ㅍ', 13: 'ㅎ',
    14: 'ㅏ', 15: 'ㅑ', 16: 'ㅓ', 17: 'ㅕ', 18: 'ㅗ', 19: 'ㅛ', 20: 'ㅜ',
    21: 'ㅠ', 22: 'ㅡ', 23: 'ㅣ', 24: 'ㅐ', 25: 'ㅒ', 26: 'ㅔ', 27: 'ㅖ',
    28: 'ㅚ', 29: 'ㅟ', 30: 'ㅢ',
    31: 'end',
    # --- 쌍자음 추가 ---
    32: 'ㄲ', 33: 'ㄸ', 34: 'ㅃ', 35: 'ㅆ', 36: 'ㅉ',
    # --- 특수 키 ---
    47: ' ', 48: 'backspace'
}

FEATURES = 63
MODEL_SAVE_PATH = "Model/hangul_chosung_model3.h5"
LABEL_MAP_SAVE_PATH = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "hangul_label_map.json")
SCALER_SAVE_PATH = os.path.join(os.path.dirname(MODEL_SAVE_PATH), "scaler.joblib")

# --- 2. 데이터 로드 (Data Loading) ---
X, y = [], []
data_counts = {k: 0 for k in HANGUL.keys()}

print("--- 데이터 로드 시작 ---")
print(f"현재 작업 디렉토리: {os.getcwd()}")
print(f"데이터 디렉토리 (설정된 경로): {DATA_DIR}")
print(f"데이터 디렉토리 (절대 경로): {os.path.abspath(DATA_DIR)}")

if not os.path.exists(DATA_DIR):
    print(f"[ERROR] 데이터 디렉토리 '{DATA_DIR}'를 찾을 수 없습니다. 경로를 확인해주세요.")
    raise SystemExit(1)

all_id_folders = os.listdir(DATA_DIR)
# 숫자인 폴더만 정렬 우선
def _sort_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, name)
all_id_folders.sort(key=_sort_key)
print(f"데이터 디렉토리 '{DATA_DIR}' 내의 항목 (폴더 이름): {all_id_folders}")

for id_str in all_id_folders:
    path = os.path.join(DATA_DIR, id_str)
    if not os.path.isdir(path):
        print(f"[DEBUG] '{path}'는 폴더가 아닙니다. 건너뜁니다.")
        continue

    try:
        label_id = int(id_str)
    except ValueError:
        print(f"[WARNING] 폴더 이름 '{id_str}'은 유효한 숫자 ID가 아닙니다. 건너뜁니다.")
        continue

    if label_id not in HANGUL:
        print(f"[WARNING] 폴더 이름 '{id_str}' (ID: {label_id})은 정의되지 않은 ID입니다. 건너뜁니다.")
        continue

    label_name = HANGUL[label_id]
    print(f"[DEBUG] 폴더 '{id_str}' (ID: {label_id}, 라벨: {label_name}) 처리 중...")

    files = [f for f in os.listdir(path) if f.endswith(".npy")]
    if not files:
        print(f"[WARNING] 폴더 '{path}'에 .npy 파일이 없습니다.")
        continue

    for fn in files:
        fpath = os.path.join(path, fn)
        try:
            arr = np.load(fpath)
        except Exception as e:
            print(f"[ERROR] 파일 '{fpath}' 로드 중 오류: {e}. 건너뜁니다.")
            continue

        if arr.shape == (FEATURES,):
            X.append(arr)
            y.append(label_id)
            data_counts[label_id] += 1
        else:
            print(f"[WARNING] '{fn}' 형상 불일치: {arr.shape} (예상: ({FEATURES},)). 건너뜁니다.")

print("\n[INFO] Class data distribution (원래 ID 기준):")
for label_id in sorted(data_counts.keys()):
    if data_counts[label_id] > 0: # 데이터가 있는 클래스만 출력
        print(f"  ID {label_id:>2} ({HANGUL.get(label_id,'?')}): {data_counts[label_id]} samples")
print("-" * 40)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("--- 데이터 로드 완료 ---")
if X.size == 0:
    print("[ERROR] 로드된 데이터가 없습니다. 학습을 시작할 수 없습니다.")
    raise SystemExit(1)

print(f"[INFO] 로드된 입력 데이터 형상 (X): {X.shape}")
print(f"[INFO] 로드된 라벨 데이터 형상 (y): {y.shape}")

# --- 2.1 라벨 연속화 (Label Remap to 0..C-1) ---
unique_ids = np.unique(y)
num_classes = len(unique_ids)
id2idx = {int(old): int(i) for i, old in enumerate(unique_ids)}
idx2id = {int(i): int(old) for i, old in enumerate(unique_ids)}
y_mapped = np.vectorize(id2idx.get)(y)

print(f"[INFO] 고유 라벨 ID들: {unique_ids.tolist()}")
print(f"[INFO] 클래스 개수(num_classes): {num_classes}")
print(f"[INFO] 재매핑 후 y 범위: [{y_mapped.min()}, {y_mapped.max()}]")

# --- 2.2 라벨맵 저장 (for inference) ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
with open(LABEL_MAP_SAVE_PATH, "w", encoding="utf-8") as f:
    json.dump({"id2idx": id2idx, "idx2id": idx2id}, f, ensure_ascii=False, indent=2)
print(f"[INFO] 라벨 맵 저장 완료: {LABEL_MAP_SAVE_PATH}")

# --- 3. 데이터 스케일링 (Scaling) ---
print("[INFO] 데이터 스케일링 시작...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("[INFO] 데이터 스케일링 완료.")
print(f"[INFO] 스케일링된 입력 데이터 형상 (X_scaled): {X_scaled.shape}")

# --- 4. 데이터 분할 (Split) ---
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
)
print(f"[INFO] 학습 데이터 형상: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"[INFO] 테스트 데이터 형상: X_test={X_test.shape}, y_test={y_test.shape}")

# --- 4.1 클래스 가중치 (선택) ---
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=y_train
)
class_weight = {int(i): float(w) for i, w in enumerate(class_weights_array)}
print(f"[INFO] 클래스 가중치 예시(일부): {dict(list(class_weight.items())[:8])} ...")

# --- 5. 모델 정의 (Model) ---
model = Sequential([
    Dense(128, activation='relu', input_shape=(FEATURES,)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 6. 학습 (Train) ---
print("--- 모델 학습 시작 ---")
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-5,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weight,
    verbose=1
)
print("--- 모델 학습 완료 ---")

# --- 7. 저장 (Save) ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"[INFO] 모델 저장 완료: {MODEL_SAVE_PATH}")

joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"[INFO] Scaler 저장 완료: {SCALER_SAVE_PATH}")

# --- 8. 시각화 (Plot) ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_results.png")
print("학습 결과 플롯이 'training_results.png' 파일로 저장되었습니다.")
plt.close()

print("\n--- 학습 완료 ---")
print("이제 저장된 모델과 라벨맵을 사용해 추론 시 예측 인덱스를 원래 ID로 복원할 수 있습니다.")
