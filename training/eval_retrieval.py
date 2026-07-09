"""Prompt-following retrieval eval: is the conditioning good enough?

Generates one grid per prompt from a battery, renders each through the
glyph atlas, scores every (grid, prompt) pair with CLIP, and reports
top-1 retrieval accuracy: the fraction of grids that score highest
with their OWN prompt. Chance is 1/N; a well-conditioned model should
be far above it. The battery has two splits so the number separates
memorization from generalization:

  seen:   subjects from the training prompt bank
  unseen: subjects that never appear in the bank (frozen-CLIP
          generalization is the claim being tested)

    python training/eval_retrieval.py \
        --checkpoint checkpoints/final_model.pt --ema --device cuda

The confusion pairs it prints (grid X scored best with prompt Y) are
the qualitative debugging companion to the single number.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W, UNMASK_STEPS

SEEN = ["a dragon", "a lighthouse", "a butterfly", "a sailboat",
        "a castle", "a fish", "a cactus", "an owl", "a windmill",
        "a rocket"]
UNSEEN = ["a submarine periscope", "a saxophone player", "a gargoyle",
          "a paper crane", "a chandelier", "a cello", "a samurai sword",
          "a watering hole", "a grand piano", "a hot air balloon ride"]


def evaluate(model, prompts, scorer, generator=None, device="cpu",
             guidance=2.0, schedule="rise", temperature=0.8,
             steps=UNMASK_STEPS, seed=0):
    """Returns (accuracy, confusions, score_matrix). scorer and
    generator are injectable for tests: scorer(grids, prompt) -> [N]
    scores; generator(model, prompt, ...) -> final grid."""
    if generator is None:
        from data.text_embed import embed_captions
        from model.inference import generate as _gen
        tokens, masks = embed_captions(prompts)

        def generator(model, prompt, toks, msk):
            torch.manual_seed(seed)
            _, final = _gen(model, GRID_H, GRID_W, num_steps=steps,
                            temperature=temperature, device=device,
                            cond_tokens=toks, cond_mask=msk,
                            guidance_scale=guidance,
                            guidance_schedule=schedule)
            return final
    else:
        # injected generators bring their own conditioning (tests)
        tokens = masks = [None] * len(prompts)
    grids = [generator(model, p, tokens[i], masks[i])
             for i, p in enumerate(prompts)]

    # Score matrix: rows = grids, cols = prompts
    scores = torch.stack([scorer(grids, p) for p in prompts], dim=1)
    best = scores.argmax(dim=1)
    correct = (best == torch.arange(len(prompts))).float()
    confusions = [(prompts[i], prompts[int(best[i])])
                  for i in range(len(prompts)) if best[i] != i]
    return float(correct.mean()), confusions, scores


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--guidance", type=float, default=2.0)
    parser.add_argument("--schedule", default="rise",
                        choices=["constant", "rise", "fall"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=UNMASK_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from probe import load_checkpoint
    from data.clip_rank import clip_scores

    model, conditioned = load_checkpoint(args.checkpoint,
                                         use_ema=args.ema,
                                         device=args.device)
    if not conditioned:
        raise SystemExit("Checkpoint is unconditioned — retrieval eval "
                         "is meaningless.")
    model.eval()

    def scorer(grids, prompt):
        return clip_scores(grids, prompt, device=args.device)

    for name, battery in [("seen", SEEN), ("unseen", UNSEEN)]:
        acc, confusions, _ = evaluate(
            model, battery, scorer, device=args.device,
            guidance=args.guidance, schedule=args.schedule,
            temperature=args.temperature, steps=args.steps,
            seed=args.seed)
        print(f"\n[{name}] top-1 retrieval accuracy: {acc:.0%} "
              f"(chance {1 / len(battery):.0%}, n={len(battery)})")
        for actual, best in confusions:
            print(f"  confused: generated '{actual}' -> scored best "
                  f"with '{best}'")


if __name__ == "__main__":
    main()
