import torch, torch.nn as nn

class LSTMGestureClassifier(nn.Module):
    def __init__(self, input_dim=63, hidden=128, num_layers=2, num_classes=8, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, num_layers=num_layers,
                            dropout=dropout, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):          # x: (B,L,63)
        out, _ = self.lstm(x)      # (B,L,2H)
        feat = out.mean(dim=1)     # (B,2H)  temporal average pooling
        logits = self.head(feat)   # (B,C)
        return logits
