"""Remap tonal-dialect grids onto a canonical density ramp.

The 6D converter picks freely among 95 glyphs, so a mid-gray cell has
half a dozen near-tied "correct" answers (o/e/q/G/4...). The model
faithfully learns that ambiguity as a probability smear — and a
confidence-ordered decoder cannot commit a smear, which presents as
wisp-collapse: prompted tonal output rendered entirely in faint
low-confidence glyphs. Classic ASCII artists constrain themselves to a
short density ramp for the same underlying reason: tone needs an
ordered, unambiguous vocabulary.

This tool rewrites every UNTAGGED (tonal-dialect) grid in a dataset
payload so each ink glyph becomes the ramp character of nearest ink
density. Per-cell entropy collapses ~10x; silhouette/outline samples
(tagged) pass through unchanged — their binarized vocabulary is
already low-entropy.

    python data/remap_tonal.py --data data/synthetic_v3.pt \
        --out data/synthetic_v3_ramp.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import VOCAB_SIZE
from data.charset import char_to_idx

# The classic ramp, light to dark. Space is intentionally NOT a target:
# ink must stay ink (the faintest strokes map to '.', not to nothing).
RAMP = ".:-=+*#%@"

_STYLE_TAGS = (", silhouette", ", outline style")


def build_ramp_lut():
    """[VOCAB_SIZE] long tensor mapping every glyph to its ramp glyph.

    Density comes from the same rendered-bitmap atlas the perceptual
    loss and the site's renderer use, so "nearest density" agrees with
    what the eye sees. PAD/MASK/space map to themselves.
    """
    from model.render import glyph_atlas
    atlas = glyph_atlas()
    if atlas.sum() == 0:
        raise RuntimeError("No font available — cannot compute glyph "
                           "densities for the remap.")
    density = atlas.reshape(VOCAB_SIZE, -1).mean(dim=1)     # [V]
    ramp_idx = torch.tensor([char_to_idx(c) for c in RAMP])
    ramp_density = density[ramp_idx]                        # [R]

    lut = torch.arange(VOCAB_SIZE)
    space = char_to_idx(" ")
    for v in range(VOCAB_SIZE):
        if v <= space:            # PAD, MASK, space stay themselves
            continue
        nearest = (ramp_density - density[v]).abs().argmin()
        lut[v] = ramp_idx[nearest]
    return lut


def remap_payload(payload, lut=None):
    """Remap tonal (untagged-caption) grids in-place; returns counts."""
    if lut is None:
        lut = build_ramp_lut()
    caps = payload.get("captions") or []
    ids = payload.get("caption_ids")
    data = payload["data"]
    remapped = 0
    for i in range(len(data)):
        cid = int(ids[i]) if ids is not None else -1
        caption = caps[cid] if 0 <= cid < len(caps) else ""
        if caption.endswith(_STYLE_TAGS):
            continue              # silhouette/outline keep their glyphs
        data[i] = lut[data[i].long()].to(data.dtype)
        remapped += 1
    return remapped, len(data)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True,
                        help="input dataset payload (.pt)")
    parser.add_argument("--out", required=True,
                        help="output path for the remapped payload")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if os.path.exists(args.out) and not args.overwrite:
        raise SystemExit(f"{args.out} already exists — pass --overwrite "
                         f"or choose another --out.")

    payload = torch.load(args.data, weights_only=True)
    n_remapped, n_total = remap_payload(payload)
    torch.save(payload, args.out)
    print(f"Remapped {n_remapped:,}/{n_total:,} grids (tonal dialect) "
          f"onto the '{RAMP}' ramp -> {args.out}")

    from data.charset import grid_to_string
    for i in range(len(payload["data"])):
        cid = int(payload["caption_ids"][i])
        cap = payload["captions"][cid] if cid >= 0 else ""
        if not cap.endswith(_STYLE_TAGS):
            print(f'\n--- sample: "{cap}" ---')
            print(grid_to_string(payload["data"][i].long()))
            break


if __name__ == "__main__":
    main()
