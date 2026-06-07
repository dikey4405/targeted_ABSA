import torch
from torch import nn

class ConditionalAttention(nn.Module):
    def __init__(self, hidden_size: int, attn_size: int = 256):
        super().__init__()
        self.query_proj = nn.Linear(hidden_size, attn_size)
        self.key_proj = nn.Linear(hidden_size, attn_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, query_vector, mask):
        query = self.query_proj(query_vector).unsqueeze(1)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        scores = torch.bmm(query, key.transpose(1, 2)).squeeze(1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), value).squeeze(1)
        return pooled
