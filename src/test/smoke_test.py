import torch
from model import BiEncoder
from losses import info_nce_loss

model = BiEncoder().cuda()
B, L = 16, 32

ids  = torch.randint(0, 1000, (B, L)).cuda()
mask = torch.ones(B, L, dtype=torch.long).cuda()

q = model(ids, mask)
d = model(ids, mask)
print(q.shape)                    # (16, 384)
print(q.norm(dim=1)[:3])          # ~1.0 olmalı
print(info_nce_loss(q, d).item()) # ln(16)=2.77 civarı değil, ~0 olmalı — neden?