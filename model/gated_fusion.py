import torch
from torch import nn

class GatedFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
    
    def forward(self, cls_vec, attn_vec):
        gate = self.gate(torch.cat([cls_vec, attn_vec], dim=-1))
        fused = gate * cls_vec + (1 - gate) * attn_vec
        return fused