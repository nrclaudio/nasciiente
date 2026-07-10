"""Coarse-to-fine probe: pin the layout small, then upscale it.

The edge-echo artifact is a parallel-commit coordination failure: on a
blank 48x80 canvas, "a rectangle" has probability mass at every
position it could legally occupy, and independently sampled commits
land several offset copies of each edge. This probe attacks the cause
instead of the symptom (Muse's answer to the same problem): generate on
a canvas with `factor**2` times fewer cells — far fewer joint decisions,
so the position ambiguity mostly collapses — then enlarge it with the
existing anchored upscale, which is a pure inpainting task the model
trains on directly (anchor masks).

Prints the coarse draft and the upscaled result side by side, so both
phases can be judged separately: a clean-but-small draft with a messy
upscale blames the upscaler; a messy draft blames coarse generation.

Usage:
    python probe_coarse.py --checkpoint checkpoints/geometry_last.pt \
        --prompt "a rectangle" --prompt "a diamond" --prompt "a cross" \
        --guidance 1.5 --schedule rise --gumbel 0 --out probe_coarse.txt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import GRID_H, GRID_W, CFG_SCALE
from data.charset import grid_to_string
from model.inference import generate, upscale_grid
from probe import load_checkpoint

DEFAULT_PROMPTS = ["a rectangle", "a diamond", "a cross"]


def ink(grid):
    return int((grid > 2).sum())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--coarse-rows", type=int, default=GRID_H // 2)
    parser.add_argument("--coarse-cols", type=int, default=GRID_W // 2)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--steps", type=int, default=32,
                        help="unmasking steps for the coarse draft")
    parser.add_argument("--upscale-steps", type=int, default=16,
                        help="unmasking steps for the upscale fill")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gumbel", type=float, default=0.0)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--schedule", default="rise",
                        choices=["constant", "rise", "fall"])
    parser.add_argument("--space-bias", type=float, default=0.0)
    parser.add_argument("--revision-steps", type=int, default=0)
    parser.add_argument("--revision-fraction", type=float, default=0.1)
    parser.add_argument("--max-commit", type=int, default=None,
                        help="cap cells committed per decode step "
                             "(sequentializes the ambiguous tail; on the "
                             "coarse canvas even 4-8 is cheap)")
    parser.add_argument("--no-cond", action="store_true",
                        help="skip CLIP; unconditional coarse draft "
                             "(mainly for smoke-testing the pipeline)")
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

    prompts = args.prompt or DEFAULT_PROMPTS
    runs = []
    if args.no_cond or not conditioned:
        runs = [("unconditional", {})]
    else:
        from data.text_embed import embed_captions
        tokens, masks = embed_captions(prompts)
        for text, toks, msk in zip(prompts, tokens, masks):
            runs.append((f'"{text}"',
                         dict(cond_tokens=toks, cond_mask=msk,
                              guidance_scale=args.guidance)))

    shared = dict(temperature=args.temperature,
                  gumbel_scale=args.gumbel,
                  guidance_schedule=args.schedule,
                  space_bias=args.space_bias,
                  revision_steps=args.revision_steps,
                  revision_fraction=args.revision_fraction,
                  max_commit=args.max_commit,
                  device=device)

    lines = []
    for label, kwargs in runs:
        torch.manual_seed(args.seed)
        _, coarse = generate(model, args.coarse_rows, args.coarse_cols,
                             num_steps=args.steps, **shared, **kwargs)
        _, big = upscale_grid(model, coarse, factor=args.factor,
                              num_steps=args.upscale_steps,
                              **shared, **kwargs)

        block = "\n".join([
            f"=== {label} · coarse {args.coarse_rows}x{args.coarse_cols} "
            f"(ink: {ink(coarse)}) ===",
            grid_to_string(coarse),
            f"--- upscaled x{args.factor} (ink: {ink(big)}) ---",
            grid_to_string(big),
        ])
        print("\n" + block)
        lines.append(block)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n\n".join(lines))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
