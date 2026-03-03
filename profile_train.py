"""Quick profiling — gradient checkpointing enabled."""
import os, sys, time, gc
sys.path.insert(0, os.path.dirname(__file__))

import torch
from config import GRID_H, GRID_W, VOCAB_SIZE

device = torch.device("mps")
print(f"Device: {device}")

from model.ascii_bert import ASCIIBert
model = ASCIIBert().to(device)
print(f"Gradient checkpointing: {model.transformer.gradient_checkpointing}")
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for bs in [1, 2, 4]:
    gc.collect()
    torch.mps.empty_cache()

    x = torch.randint(0, VOCAB_SIZE, (bs, GRID_H, GRID_W)).to(device)
    target = torch.randint(2, VOCAB_SIZE, (bs, GRID_H, GRID_W)).to(device)
    mask = (torch.rand(bs, GRID_H, GRID_W) > 0.5).to(device)

    model.train()
    optimizer.zero_grad()

    try:
        t0 = time.time()
        logits = model(x)
        loss = model.compute_loss(logits, target, mask)
        loss.backward()
        optimizer.step()
        torch.mps.synchronize()
        elapsed = time.time() - t0
        print(f"B={bs}: {elapsed:.2f}s, loss={loss.item():.4f}")
    except RuntimeError as e:
        print(f"B={bs}: FAILED — {e}")
        break

    del x, target, mask, logits, loss

print("\nDone.")
