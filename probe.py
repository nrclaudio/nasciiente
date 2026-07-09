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


def load_checkpoint(path, use_ema=False, device="cpu"):
    """Build a model shaped like the checkpoint (any --model-size) and
    load it. Returns (model, conditioned)."""
    from model.ascii_bert import load_compatible_state, \
        model_matching_state
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
    model = model_matching_state(state).to(device)
    try:
        return model, load_compatible_state(model, state)
    except ValueError as e:
        raise SystemExit(f"{path}: {e}")


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
    parser.add_argument("--space-bias", type=float, default=0.0,
                        help="anti-blank pressure (try 2-6 when everything "
                             "generates empty); annealed with mask ratio")
    parser.add_argument("--gumbel", type=float, default=1.0,
                        help="exploration noise on commit ORDER (0 = "
                             "strictly most-confident-first; the "
                             "default 1.0 randomizes early commits for "
                             "diversity, which hurts precise structure)")
    parser.add_argument("--revision-steps", type=int, default=2)
    parser.add_argument("--revision-fraction", type=float, default=0.1)
    parser.add_argument("--cluster-conf", action="store_true",
                        help="rank decode commitment by visual-cluster "
                             "probability mass instead of single-glyph "
                             "probability — counters wisp-collapse on "
                             "tonal checkpoints, keeps all 95 glyphs")
    parser.add_argument("--schedule", default="constant",
                        choices=["constant", "rise", "fall"],
                        help="CFG schedule over the decode: 'rise' grows "
                             "guidance from 1 to the full scale as cells "
                             "commit (counters the early ink flood at "
                             "scale >= 2); A/B against 'constant'")
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
    model, conditioned = load_checkpoint(args.checkpoint,
                                         use_ema=args.ema, device=device)
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
        from data.text_embed import embed_captions
        tokens, masks = embed_captions(prompts)
        for text, toks, msk in zip(prompts, tokens, masks):
            for g in args.guidance:
                runs.append((f'"{text}" @ guidance {g:g}',
                             dict(cond_tokens=toks, cond_mask=msk,
                                  guidance_scale=g)))

    lines = []
    for label, kwargs in runs:
        torch.manual_seed(args.seed)
        _, grid = generate(model, args.rows, args.cols,
                           num_steps=args.steps,
                           temperature=args.temperature,
                           space_bias=args.space_bias,
                           guidance_schedule=args.schedule,
                           cluster_confidence=args.cluster_conf,
                           gumbel_scale=args.gumbel,
                           revision_steps=args.revision_steps,
                           revision_fraction=args.revision_fraction,
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
