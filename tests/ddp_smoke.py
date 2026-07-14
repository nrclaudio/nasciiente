"""2-process CPU (gloo) DDP smoke test for the conditioned training path.

Run via: python -m torch.distributed.run --standalone --nproc_per_node=2 \
             tests/ddp_smoke.py
(test_ddp.py does this automatically.)

Covers the failure modes a single-process test can't:
- a fully-captioned batch with cond_drop all False — every parameter
  (incl. the null token) must still participate in autograd or DDP's
  reducer errors out mid-run;
- a mixed captioned/uncaptioned batch;
- replicas staying bit-identical after optimizer steps.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.ascii_bert import ASCIIBert


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)  # same init everywhere (DDP broadcasts anyway)

    model = DDP(ASCIIBert(embed_dim=32, num_layers=2, num_heads=2,
                          ffn_dim=64, text_dim=16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    torch.manual_seed(100 + rank)  # per-rank data, like the real loader
    h, w, B = 12, 20, 4
    toks = torch.randn(B, 6, 16)
    msk = torch.ones(B, 6, dtype=torch.bool)

    batches = [
        # (cond_tokens, cond_mask, cond_drop) — the all-False-drop batch
        # is the DDP trap: null_token selected nowhere, must still get a
        # (zero) gradient
        (toks, msk, torch.zeros(B, dtype=torch.bool)),
        # mixed: some rows dropped to null
        (toks, msk, torch.tensor([True, False, True, False])),
        # unconditional batch
        (None, None, None),
    ]
    for cond_tokens, cond_mask, cond_drop in batches:
        x = torch.randint(2, 98, (B, h, w))
        target = torch.randint(2, 98, (B, h, w))
        mask = torch.rand(B, h, w) > 0.5
        ratio = torch.rand(B)
        # Mirror the real train_epoch: the critic head joins EVERY
        # forward/loss — this smoke caught the reducer crash when a
        # forward without it left critic.{weight,bias} gradient-less
        # (the 14.4 conditionally-used-parameter trap, second edition)
        logits, critic = model(x, cond_tokens=cond_tokens,
                               cond_mask=cond_mask, cond_drop=cond_drop,
                               mask_ratio=ratio, return_critic=True)
        loss = model.module.compute_loss(logits, target, mask)
        visible = torch.ones_like(x, dtype=torch.bool)
        critic_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            critic[visible].float(), (x == target)[visible].float())
        loss = loss + 0.1 * critic_bce
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Replicas must be bit-identical after synced steps
    for name, p in model.module.named_parameters():
        others = [torch.empty_like(p) for _ in range(dist.get_world_size())]
        dist.all_gather(others, p.detach())
        for other in others:
            assert torch.equal(other, p.detach()), f"replica drift: {name}"

    dist.barrier()
    if rank == 0:
        print("DDP SMOKE OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
