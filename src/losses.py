import torch
import torch.nn.functional as F

def info_nce_loss(q_emb, d_emb, temperature=0.05):
    scores = q_emb @ d_emb.T
    scores = scores / temperature
    labels = torch.arange(len(scores), device=scores.device)
    return F.cross_entropy(scores, labels)