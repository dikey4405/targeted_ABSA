import torch
from torch import nn

class MLPHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, dropout: float = 0.2, mid_size: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)