import torch
from torch import nn
class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int, attn_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, attn_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_size, 1),
        )
    
    def forward(self, hidden_states, mask):
        scores = self.proj(hidden_states).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled

