# src/CTC/ctc_model.py

import torch
import torch.nn as nn

class GestureCTCNet(nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=10):
        super(GestureCTCNet, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes + 1)  

    def forward(self, x):
        x, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        x = self.classifier(x)  # [batch, seq_len, num_classes+1]
        x = x.permute(1, 0, 2)  # [seq_len, batch, num_classes+1] → for CTC Loss
        return x
