import os, numpy as np, torch
from torch.utils.data import Dataset

TARGET_WORDS = ["arrival","correct","delete","description","emergency","start","traffic","voice"]

def hand_center_scale_norm(seq_t63: np.ndarray) -> np.ndarray:
    T = seq_t63.shape[0]
    seq = seq_t63.reshape(T,21,3).copy()
    wrist = seq[:,0:1,:]
    seq -= wrist
    xy = seq[:,:,:2]
    min_xy = xy.min(axis=1, keepdims=True)
    max_xy = xy.max(axis=1, keepdims=True)
    diag = np.linalg.norm(max_xy - min_xy, axis=2, keepdims=True)
    diag[diag < 1e-6] = 1.0
    seq /= diag
    return seq.reshape(T,63).astype(np.float32)

def time_resample(seq_t63: np.ndarray, fixed_len: int) -> np.ndarray:
    T, C = seq_t63.shape
    if T == fixed_len: return seq_t63
    src = np.linspace(0, T-1, num=T)
    dst = np.linspace(0, T-1, num=fixed_len)
    out = np.empty((fixed_len, C), dtype=np.float32)
    for i in range(C): out[:, i] = np.interp(dst, src, seq_t63[:, i])
    return out

class GestureClsDataset(Dataset):
    """
    dataset/
      arrival/*.npy
      ...
    라벨: 단어별 0..N-1
    """
    def __init__(self, root_dir: str, fixed_len: int = 100):
        self.root = root_dir
        self.words = [w for w in TARGET_WORDS if os.path.isdir(os.path.join(root_dir, w))]
        self.words.sort()
        self.word_to_id = {w:i for i,w in enumerate(self.words)}
        self.id_to_word = {i:w for w,i in self.word_to_id.items()}
        self.fixed_len = fixed_len

        self.files, self.labels = [], []
        for w in self.words:
            wdir = os.path.join(root_dir, w)
            for fn in os.listdir(wdir):
                if fn.endswith(".npy"):
                    self.files.append(os.path.join(wdir, fn))
                    self.labels.append(self.word_to_id[w])

        print(f"[Dataset] classes={self.words}  samples={len(self.files)}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y = self.labels[idx]
        x = np.load(path).astype(np.float32)   # (T,63)
        x = hand_center_scale_norm(x)
        x = time_resample(x, self.fixed_len)   # (L,63), L=fixed_len
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
