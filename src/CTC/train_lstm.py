import os, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from cls_dataset import GestureClsDataset
from lstm_classifier import LSTMGestureClassifier

ROOT = Path(__file__).resolve().parents[2]    
DATA = ROOT/"dataset"
SAVE = Path(__file__).resolve().parent            
DEVICE = torch.device("cpu")

EPOCHS=25; BATCH=64; LR=2e-4; FIXED_T=100; HIDDEN=128

# dataset + split
full = GestureClsDataset(str(DATA), fixed_len=FIXED_T)
idx = np.random.RandomState(42).permutation(len(full))
n_val = max(1, int(0.2*len(full)))
val_idx, tr_idx = idx[:n_val], idx[n_val:]

class _Sub(torch.utils.data.Dataset):
    def __init__(self, ds, idxs): self.ds, self.idxs = ds, idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i): return self.ds[self.idxs[i]]

train_ds = _Sub(full, tr_idx); val_ds = _Sub(full, val_idx)

# class-balanced sampler
counts = {i:0 for i in full.id_to_word.keys()}
for i in tr_idx:
    _, y = full[i]; counts[int(y.item())]+=1
total = sum(counts.values())
w_class = {k: total/max(1,c) for k,c in counts.items()}
weights = []
for i in tr_idx:
    _, y = full[i]; weights.append(w_class[int(y.item())])
sampler = WeightedRandomSampler(weights, num_samples=len(tr_idx), replacement=True)

def collate(batch):
    xs = torch.stack([b[0] for b in batch])   # (B,L,63)
    ys = torch.stack([b[1] for b in batch])   # (B,)
    return xs, ys

train_ld = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, collate_fn=collate)
val_ld   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)

model = LSTMGestureClassifier(input_dim=63, hidden=HIDDEN, num_layers=2,
                              num_classes=len(full.words), dropout=0.2).to(DEVICE)
# label smoothing 살짝
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

def evaluate(loader, name="val"):
    model.eval()
    tot=0; correct=0
    per_cls = {i: {"tp":0,"tot":0} for i in full.id_to_word.keys()}
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(DEVICE))
            pred = logits.argmax(dim=1).cpu()
            y = yb.cpu()
            tot += len(y); correct += int((pred==y).sum())
            for yi, pi in zip(y.tolist(), pred.tolist()):
                per_cls[yi]["tot"]+=1
                if yi==pi: per_cls[yi]["tp"]+=1
    acc = correct/max(1,tot)
    print(f"[{name}] acc={acc*100:.2f}% ({correct}/{tot})")
    for k in sorted(per_cls.keys()):
        t=per_cls[k]["tot"]; tp=per_cls[k]["tp"]
        if t>0:
            print(f"  - {k:2d} {full.id_to_word[k]:12s}: {tp}/{t} = {tp/max(1,t):.2f}")
    return acc

best= -1
for epoch in range(1, EPOCHS+1):
    model.train()
    running=0.0
    for step,(xb,yb) in enumerate(train_ld,1):
        logits = model(xb.to(DEVICE))
        loss = criterion(logits, yb.to(DEVICE))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        running += loss.item()
        if step % 30 == 0:
            print(f"Epoch {epoch} Step {step}/{len(train_ld)} Loss {loss.item():.4f}")
    print(f"Epoch {epoch} AvgLoss {running/max(1,len(train_ld)):.4f}")
    evaluate(train_ld,"train")
    val_acc = evaluate(val_ld,"val")
    if val_acc > best:
        best = val_acc
        torch.save(model.state_dict(), str(SAVE/"gesture_lstm_cls.pth"))
        meta = {
            "words": full.words,
            "word_to_id": full.word_to_id,
            "id_to_word": {str(k):v for k,v in full.id_to_word.items()},
            "fixed_len": FIXED_T,
            "hidden": HIDDEN
        }
        import json; json.dump(meta, open(SAVE/"gesture_lstm_cls.json","w"), ensure_ascii=False, indent=2)
        print(f"[SAVE] best={best:.4f} -> gesture_lstm_cls.pth / gesture_lstm_cls.json")
print("[DONE] training")
