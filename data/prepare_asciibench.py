"""Convert the ASCIIBench dataset into a nASCIIente training payload.

ASCIIBench ("ASCIIBench: Evaluating Language-Model-Based Understanding of
Visually-Oriented Text", arXiv 2512.04125; github.com/KerryLuo/ASCIIBench)
ships ~5.1k class-labeled human-made ASCII pieces as final_dataset.jsonl
with records {class, unique_id, file_name, ascii_art}. ~97% fit the 48x80
canvas directly — roughly doubling the human-art corpus for the stage-3
fine-tune, with cleaner labels than BLIP captions (752 human-assigned
classes).

NOTE the repo posts the file without an explicit license (README says
"under construction"). Fine for local training experiments; do NOT commit
the jsonl or the converted payload to a public repo, and check with the
authors (contact in their README) before redistributing.

Usage:
    curl -sLO https://raw.githubusercontent.com/KerryLuo/ASCIIBench/main/final_dataset.jsonl
    python data/prepare_asciibench.py --jsonl final_dataset.jsonl
    # -> data/asciibench_data.pt, loadable anywhere human_data.pt is
    #    (e.g. point the human stage at it, or list it as extra replay)
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.charset import grid_to_string
from data.prepare_human_ascii import text_to_training_grid
from data.prompt_bank import _article


def class_to_caption(name):
    """'omega' -> 'an omega'; 'sea_creature' -> 'a sea creature'.

    Class names are single nouns or category words; the article form
    matches the engine's caption style so the pieces reinforce (rather
    than fork) the prompt distribution.
    """
    name = name.replace("_", " ").replace("-", " ").strip().lower()
    return _article(name) if name else None


def convert(jsonl_path):
    """Returns (grids [N,48,80] uint8, caption_ids, captions, stats)."""
    grids, caption_ids = [], []
    caption_index = {}
    skipped_fit = skipped_caption = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            grid = text_to_training_grid(rec.get("ascii_art", ""))
            if grid is None:
                skipped_fit += 1
                continue
            caption = class_to_caption(rec.get("class", ""))
            if not caption:
                skipped_caption += 1
                continue
            grids.append(grid.to(torch.uint8))
            caption_ids.append(caption_index.setdefault(caption,
                                                        len(caption_index)))
    stats = dict(kept=len(grids), skipped_fit=skipped_fit,
                 skipped_caption=skipped_caption,
                 classes=len(caption_index))
    return (torch.stack(grids) if grids else torch.empty(0),
            torch.tensor(caption_ids, dtype=torch.long),
            list(caption_index), stats)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl", required=True,
                        help="path to ASCIIBench final_dataset.jsonl")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "asciibench_data.pt"))
    args = parser.parse_args()

    data, caption_ids, captions, stats = convert(args.jsonl)
    if not len(data):
        raise SystemExit("no usable records — wrong file?")
    torch.save({"data": data, "caption_ids": caption_ids,
                "captions": captions}, args.out)
    print(f"Kept {stats['kept']:,} pieces across {stats['classes']} classes "
          f"(skipped: {stats['skipped_fit']} didn't fit/too sparse, "
          f"{stats['skipped_caption']} unusable class labels)")
    print(f"Saved to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print("\n=== Samples ===")
    for i in range(0, min(len(data), 2)):
        print(f'--- "{captions[caption_ids[i]]}" ---')
        print(grid_to_string(data[i].long()))


if __name__ == "__main__":
    main()
