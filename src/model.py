import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

def mean_pooling(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).float()
    numerator     = (last_hidden_state * mask_expanded).sum(dim=1)
    denominator   = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return numerator / denominator


class BiEncoder(nn.Module):
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # ← senin yazacağın 3 satır
        ...