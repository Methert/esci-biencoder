import torch
import torch.nn.functional as F
print(torch.__version__)
print(torch.cuda.is_available())

x = torch.arange(24, dtype=torch.float).reshape(2, 4, 3)
print(x.shape)

def mean_pooling(last_hidden_state, attention_mask):
    """
    last_hidden_state: (B, L, H)
    attention_mask:    (B, L)
    returns:           (B, H)
    """
    mask_expanded = attention_mask.unsqueeze(-1).float()
    print(mask_expanded.shape)    # (2,4,1) bekleniyor
    numerator     = (last_hidden_state * mask_expanded).sum(dim=1)
    print(numerator.shape) 
    denominator   = mask_expanded.sum(dim=1).clamp(min=1e-9)
    print(denominator.shape)      # (2,1)
    return numerator / denominator

mask = torch.tensor([[1, 1, 0, 0],
                     [1, 1, 1, 0]])
print(mean_pooling(x, mask))



embeddings = F.normalize(embeddings, p=2, dim=1)