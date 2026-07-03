"""Download and normalize human-made ASCII art into training grids.

Pulls a HuggingFace dataset of human-drawn ASCII art (default:
apehex/ascii-art, ~5.3k pieces scraped from asciiart.eu), filters to the
model's 95-char printable-ASCII vocabulary, and centers each piece on a
48x80 grid padded with spaces. Pieces that are too large, too sparse, or
contain non-ASCII characters are skipped.

Produces data/human_data.pt — used as an optional third training stage.

Usage:
    python data/prepare_human_ascii.py
    python data/prepare_human_ascii.py --dataset apehex/ascii-art --min-ink 20
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W
from data.charset import char_to_idx, grid_to_string

SPACE = char_to_idx(" ")

# Rejection thresholds
MIN_ROWS = 2
MIN_INK_CHARS = 20  # non-space characters


def text_to_training_grid(text, grid_h=GRID_H, grid_w=GRID_W,
                          min_ink=MIN_INK_CHARS):
    """Normalize one piece of ASCII art to a [grid_h, grid_w] long tensor.

    Returns None if the piece doesn't fit the grid, contains characters
    outside printable ASCII, or has too little content to learn from.
    """
    text = (text.replace("\r\n", "\n").replace("\r", "\n")
                .replace("\t", "    "))
    lines = [line.rstrip() for line in text.split("\n")]

    # Trim leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not (MIN_ROWS <= len(lines) <= grid_h):
        return None
    width = max(len(line) for line in lines)
    if width == 0 or width > grid_w:
        return None
    for line in lines:
        for ch in line:
            if not (32 <= ord(ch) <= 126):
                return None
    ink = sum(len(line.replace(" ", "")) for line in lines)
    if ink < min_ink:
        return None

    grid = torch.full((grid_h, grid_w), SPACE, dtype=torch.long)
    top = (grid_h - len(lines)) // 2
    left = (grid_w - width) // 2
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            grid[top + r, left + c] = char_to_idx(ch)
    return grid


def extract_text(record):
    """Find the ASCII art string in a dataset record."""
    for key in ("content", "text", "art", "ascii", "output", "completion"):
        value = record.get(key)
        if isinstance(value, str) and "\n" in value:
            return value
    # Fallback: any multi-line string field
    for value in record.values():
        if isinstance(value, str) and "\n" in value:
            return value
    return None


def load_human_dataset(name, config=None):
    """Load a HF dataset of ASCII art, trying known config names."""
    from datasets import load_dataset

    attempts = [config] if config else ["asciiart", None]
    last_err = None
    for cfg in attempts:
        try:
            if cfg:
                ds = load_dataset(name, cfg, split="train")
                print(f"Loaded {name} (config: {cfg}): {len(ds):,} records")
            else:
                ds = load_dataset(name, split="train")
                print(f"Loaded {name}: {len(ds):,} records")
            return ds
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load {name}: {last_err}")


def main():
    parser = argparse.ArgumentParser(
        description="Normalize human ASCII art into 48x80 training grids.")
    parser.add_argument("--dataset", default="apehex/ascii-art",
                        help="HuggingFace dataset id (default: apehex/ascii-art)")
    parser.add_argument("--config", default=None,
                        help="Dataset config/subset name (default: try 'asciiart', then none)")
    parser.add_argument("--min-ink", type=int, default=MIN_INK_CHARS,
                        help="Minimum non-space characters per piece")
    args = parser.parse_args()

    ds = load_human_dataset(args.dataset, args.config)

    grids = []
    rejected = {"no_text": 0, "size": 0}
    for record in ds:
        text = extract_text(record)
        if text is None:
            rejected["no_text"] += 1
            continue
        grid = text_to_training_grid(text, min_ink=args.min_ink)
        if grid is None:
            rejected["size"] += 1
            continue
        grids.append(grid)

    print(f"Kept {len(grids):,} pieces "
          f"(rejected: {rejected['no_text']} without text, "
          f"{rejected['size']} wrong size/charset/too sparse)")

    if not grids:
        print("ERROR: No usable ASCII art found.")
        sys.exit(1)

    data = torch.stack(grids).to(torch.uint8)  # vocab fits in a byte
    out_path = os.path.join(os.path.dirname(__file__), "human_data.pt")
    torch.save(data, out_path)
    print(f"Saved {data.shape} to {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("\n=== Sample Human ASCII Art (normalized) ===\n")
    for i in range(min(3, len(grids))):
        print(f"--- Sample {i} ---")
        print(grid_to_string(grids[i]))
        print()


if __name__ == "__main__":
    main()
