"""Probe a training checkpoint with prompts — without disturbing training.

Loads a checkpoint on CPU (by default) and generates grids for a set of
prompts at one or more guidance scales, plus an unconditional baseline.
Prints each grid with an ink count so conditioning progress is measurable
at a glance. Safe to run against {stage}_last.pt while training continues
on the GPUs.

Usage:
    python probe.py --checkpoint checkpoints/geometry_last.pt
    python probe.py --checkpoint checkpoints/geometry_last.pt \
        --prompt "a rectangle" --prompt "a diamond" --guidance 1 3 6
    python probe.py --checkpoint checkpoints/shading_last.pt \
        --prompt "a cat" --ema --out probe_results.txt

What to look for:
  - unconditional blank but prompted structured -> conditioning works
    (unconditional only sees ~10% of training on captioned data)
  - different prompts -> recognizably different outputs = the model is
    learning caption semantics, not just "draw something"
  - higher guidance -> more ink but also more clutter; the sweet spot
    drops toward ~3 as training progresses
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE, CFG_SCALE
from data.charset import grid_to_string
from model.ascii_bert import ASCIIBert
from model.inference import generate

DEFAULT_PROMPTS = [
    "a rectangle",
    "a cross",
    "a diamond",
    "a rectangle and a cross",
]


def load_checkpoint(model, path, use_ema=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and use_ema:
        if "ema_state_dict" not in ckpt:
            raise SystemExit(f"{path} has no ema_state_dict — rerun "
                             f"without --ema")
        state = ckpt["ema_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = ([k for k in missing if not k.startswith("conditioning.")]
           + [k for k in unexpected if not k.startswith("conditioning.")])
    # EMA shadows only track trainable params; buffers may be "missing"
    if use_ema:
        bad = [k for k in bad if k in unexpected]
    if bad:
        raise SystemExit(f"Checkpoint {path} does not match the model "
                         f"(mismatched keys: {bad[:5]}...)")
    conditioned = not any(k.startswith("conditioning.") for k in missing)
    return conditioned


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="e.g. checkpoints/geometry_last.pt")
    parser.add_argument("--prompt", action="append", default=None,
                        help="prompt to probe (repeatable; default: a "
                             "small geometry set)")
    parser.add_argument("--guidance", type=float, nargs="+",
                        default=[CFG_SCALE],
                        help=f"guidance scale(s) (default: {CFG_SCALE})")
    parser.add_argument("--steps", type=int, default=UNMASK_STEPS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--rows", type=int, default=GRID_H)
    parser.add_argument("--cols", type=int, default=GRID_W)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed so epoch-to-epoch comparisons are "
                             "apples to apples")
    parser.add_argument("--device", default="cpu",
                        help="cpu (default; leaves training GPUs alone) "
                             "or e.g. cuda:0")
    parser.add_argument("--ema", action="store_true",
                        help="probe the EMA weights instead of the live "
                             "training weights")
    parser.add_argument("--no-unconditional", action="store_true",
                        help="skip the unconditional baseline")
    parser.add_argument("--out", default=None,
                        help="also write the results to this file")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ASCIIBert().to(device)
    conditioned = load_checkpoint(model, args.checkpoint, use_ema=args.ema)
    model.eval()
    cond_note = ("conditioned" if conditioned
                 else "UNCONDITIONED — prompts will have no effect")
    print(f"Loaded {args.checkpoint} "
          f"({'EMA' if args.ema else 'live'} weights, {cond_note})")

    prompts = args.prompt or DEFAULT_PROMPTS
    runs = []
    if not args.no_unconditional:
        runs.append(("unconditional", {}))
    if conditioned:
        # Embed all prompts in one encoder call
        from data.text_embed import embed_texts
        embs = embed_texts(prompts)
        for text, emb in zip(prompts, embs):
            for g in args.guidance:
                runs.append((f'"{text}" @ guidance {g:g}',
                             dict(cond_emb=emb, guidance_scale=g)))

    lines = []
    for label, kwargs in runs:
        torch.manual_seed(args.seed)
        _, grid = generate(model, args.rows, args.cols,
                           num_steps=args.steps,
                           temperature=args.temperature,
                           device=device, **kwargs)
        ink = int((grid > 2).sum())  # non-space printable cells
        chars = int(torch.unique(grid).numel())
        header = f"=== {label}  (ink: {ink}, unique chars: {chars}) ==="
        block = header + "\n" + grid_to_string(grid)
        print("\n" + block)
        lines.append(block)

    if args.out:
        with open(args.out, "w") as f:
            f.write("\n\n".join(lines))
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
