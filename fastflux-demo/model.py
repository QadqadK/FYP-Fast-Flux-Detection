import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True)
        self.fc = nn.Linear(h, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
