import argparse
import math
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import make_loader
from losses import info_nce_loss
from model import BiEncoder

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=None, help="stop early (smoke run)")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--out_dir", default=str(ROOT / "checkpoints"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean val InfoNCE loss + in-batch top-1 accuracy."""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for q, d in loader:
        q, d = to_device(q, device), to_device(d, device)
        with torch.autocast("cuda", dtype=torch.float16):
            q_emb, d_emb = model(**q), model(**d)
            loss = info_nce_loss(q_emb, d_emb)
        labels = torch.arange(len(q_emb), device=device)
        total_acc += ((q_emb @ d_emb.T).argmax(dim=1) == labels).float().mean().item()
        total_loss += loss.item()
        n += 1
    model.train()
    return total_loss / n, total_acc / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = make_loader(ROOT / "data" / "train.parquet", tokenizer,
                               args.batch_size, args.max_len, shuffle=True)
    val_loader = make_loader(ROOT / "data" / "val.parquet", tokenizer,
                             args.batch_size, args.max_len, shuffle=False)

    model = BiEncoder(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.max_steps or args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.amp.GradScaler("cuda")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"[zero-shot] val_loss={val_loss:.4f}  val_acc@1={val_acc:.3f}")
    best_val = val_loss

    model.train()
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        for q, d in train_loader:
            q, d = to_device(q, device), to_device(d, device)

            with torch.autocast("cuda", dtype=torch.float16):
                loss = info_nce_loss(model(**q), model(**d))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1

            if step == 1:
                print(f"[step 1] loss={loss.item():.4f}  ln(B)={math.log(args.batch_size):.4f}")
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total_steps}  loss={loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  {step / (time.time() - t0):.1f} it/s")
            if step >= total_steps:
                break

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch}] val_loss={val_loss:.4f}  val_acc@1={val_acc:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")
            print(f"  saved {out_dir / 'best.pt'}")
        if step >= total_steps:
            break

    print(f"done in {(time.time() - t0) / 60:.1f} min, peak VRAM "
          f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
