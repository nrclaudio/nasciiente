"""Inpainting probe: does the model KNOW shapes when context pins them?

The discriminating experiment for the parallel-commit (edge echo)
question. Free generation from a blank canvas confounds two abilities:
knowing the shape distribution, and coordinating thousands of parallel
commit decisions. This probe removes the second: it synthesizes a
ground-truth geometry grid, hides part of it, and asks the model to
complete it — the visible half pins the position and scale exactly.

  - Completion clean and accurate  -> the model has learned the
    distribution; echo is purely a decode-coordination problem.
  - Completion echoed/thatched even here -> the model itself is off and
    more decode engineering won't save it.

Reports per-shape masked-cell accuracy plus ink precision/recall
against the ground truth, so the verdict is a number, not a squint.

Usage:
    python probe_inpaint.py --checkpoint checkpoints/geometry_last.pt
    python probe_inpaint.py --checkpoint checkpoints/geometry_last.pt \
        --mask right --gumbel 0 --steps 32 --out probe_inpaint.txt
"""

import argparse
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import GRID_H, GRID_W
from data.charset import MASK_TOKEN, grid_to_string
from data.generate_geometry import (draw_rectangle, draw_diamond,
                                    draw_cross, draw_triangle, _blank_grid)
from model.inference import generate
from probe import load_checkpoint

SHAPES = {
    "rectangle": draw_rectangle,
    "diamond": draw_diamond,
    "cross": draw_cross,
    "triangle": draw_triangle,
}


def make_mask(mode, H, W, rng):
    """Boolean [H, W] mask of cells to HIDE."""
    m = torch.zeros(H, W, dtype=torch.bool)
    if mode == "right":
        m[:, W // 2:] = True
    elif mode == "bottom":
        m[H // 2:, :] = True
    elif mode == "block":
        bh, bw = H // 2, W // 2
        r0 = rng.randint(H // 4, H - bh)
        c0 = rng.randint(W // 4, W - bw)
        m[r0:r0 + bh, c0:c0 + bw] = True
    elif mode == "random":
        m = torch.rand(H, W) < 0.5
    else:
        raise SystemExit(f"unknown mask mode {mode}")
    return m


def score(result, truth, mask):
    """(masked-cell accuracy, ink precision, ink recall) on hidden cells."""
    res, tru = result[mask], truth[mask]
    acc = (res == tru).float().mean().item()
    ink_true = tru > 2
    ink_pred = res > 2
    tp = (ink_true & ink_pred).sum().item()
    prec = tp / max(1, ink_pred.sum().item())
    rec = tp / max(1, ink_true.sum().item())
    return acc, prec, rec


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--shape", action="append", default=None,
                        choices=list(SHAPES), help="repeatable; default all")
    parser.add_argument("--mask", default="right",
                        choices=["right", "bottom", "block", "random"],
                        help="which part of the truth grid to hide")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gumbel", type=float, default=0.0)
    parser.add_argument("--revision-steps", type=int, default=0,
                        help="default 0: measure the raw fill, no cleanup")
    parser.add_argument("--revision-fraction", type=float, default=0.1)
    parser.add_argument("--cond", action="store_true",
                        help="also pass the shape caption through CLIP "
                             "(default: unconditional — the visible half "
                             "should be context enough)")
    parser.add_argument("--guidance", type=float, default=1.5,
                        help="only used with --cond")
    parser.add_argument("--schedule", default="rise",
                        choices=["constant", "rise", "fall"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, conditioned = load_checkpoint(args.checkpoint,
                                         use_ema=args.ema, device=device)
    model.eval()
    print(f"Loaded {args.checkpoint} "
          f"({'EMA' if args.ema else 'live'} weights)")
    if args.cond and not conditioned:
        raise SystemExit("--cond requested but checkpoint is unconditioned")

    shapes = args.shape or list(SHAPES)
    lines = []
    for name in shapes:
        rng = random.Random(args.seed)
        random.seed(args.seed)  # draw_* functions use the global RNG
        truth = _blank_grid()
        SHAPES[name](truth)

        hide = make_mask(args.mask, GRID_H, GRID_W, rng)
        masked = truth.clone()
        masked[hide] = MASK_TOKEN

        kwargs = {}
        if args.cond:
            from data.text_embed import embed_captions
            toks, msks = embed_captions([f"a {name}"])
            kwargs = dict(cond_tokens=toks[0], cond_mask=msks[0],
                          guidance_scale=args.guidance)

        torch.manual_seed(args.seed)
        _, result = generate(model, GRID_H, GRID_W,
                             num_steps=args.steps,
                             temperature=args.temperature,
                             initial_grid=masked,
                             gumbel_scale=args.gumbel,
                             revision_steps=args.revision_steps,
                             revision_fraction=args.revision_fraction,
                             guidance_schedule=args.schedule,
                             device=device, **kwargs)

        acc, prec, rec = score(result, truth, hide)
        header = (f"=== {name} · mask={args.mask} · "
                  f"masked-cell acc {acc:.1%} · "
                  f"ink precision {prec:.1%} / recall {rec:.1%} ===")
        block = "\n".join([
            header,
            "--- truth ---", grid_to_string(truth),
            "--- model input ('?' = hidden) ---", grid_to_string(masked),
            "--- completion ---", grid_to_string(result),
        ])
        print("\n" + block)
        lines.append(block)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n\n".join(lines))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
