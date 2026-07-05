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


def extract_caption(record):
    """Find a short caption/title for a piece, or None.

    These are the text-conditioning signal for stage 3 — what a user's
    prompt will look like — so keep them whenever the dataset has them.
    """
    for key in ("caption", "title", "name", "subject", "category", "label",
                "prompt", "description"):
        value = record.get(key)
        if isinstance(value, str):
            value = value.strip()
            # single-line, plausibly a title (not another art blob)
            if value and "\n" not in value and len(value) <= 100:
                return value
    return None


# Boilerplate BLIP prefixes that would drown the actual subject
_CAPTION_PREFIXES = (
    "a black and white drawing of ", "a black and white photo of ",
    "a drawing of ", "a sketch of ", "a picture of ", "an image of ",
    "a photo of ", "ascii art of ",
)


def _clean_caption(text):
    text = " ".join(text.strip().split())
    lowered = text.lower()
    for prefix in _CAPTION_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            lowered = text.lower()
    return text.strip() or None


def auto_caption_grids(grids, batch_size=16, device=None, captioner=None):
    """Caption rendered grids with a local BLIP model.

    Renders each piece to pixels and asks an image captioner what it
    depicts — turning the ~60% of human pieces that have no title into
    captioned training data. `captioner` is injectable for tests: a
    callable (list of PIL images) -> list of caption strings.

    Requires: pip install transformers (BLIP downloads ~1GB once).
    """
    import torch as _torch

    from model.render import render_to_pil

    if captioner is None:
        from transformers import (BlipForConditionalGeneration,
                                  BlipProcessor)
        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        model_id = "Salesforce/blip-image-captioning-base"
        print(f"Loading {model_id} on {device} for auto-captioning...")
        processor = BlipProcessor.from_pretrained(model_id)
        blip = (BlipForConditionalGeneration.from_pretrained(model_id)
                .to(device).eval())

        @_torch.no_grad()
        def captioner(images):
            inputs = processor(images=images,
                               return_tensors="pt").to(device)
            out = blip.generate(**inputs, max_new_tokens=20)
            return processor.batch_decode(out, skip_special_tokens=True)

    captions = []
    for i in range(0, len(grids), batch_size):
        images = [render_to_pil(g) for g in grids[i:i + batch_size]]
        captions.extend(_clean_caption(c) for c in captioner(images))
        if (i // batch_size) % 10 == 0:
            print(f"  captioned {min(i + batch_size, len(grids)):,}"
                  f"/{len(grids):,}", flush=True)
    return captions


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
    parser.add_argument("--auto-caption", action="store_true",
                        help="Caption pieces WITHOUT a title using a local "
                             "BLIP model on their rendered pixels "
                             "(needs transformers; ~1GB download)")
    args = parser.parse_args()

    ds = load_human_dataset(args.dataset, args.config)

    grids = []
    caption_index = {}
    caption_ids = []
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
        caption = extract_caption(record)
        caption_ids.append(-1 if caption is None
                           else caption_index.setdefault(caption,
                                                         len(caption_index)))

    n_captioned = sum(1 for c in caption_ids if c >= 0)
    print(f"Kept {len(grids):,} pieces "
          f"({n_captioned:,} with captions; "
          f"rejected: {rejected['no_text']} without text, "
          f"{rejected['size']} wrong size/charset/too sparse)")

    if not grids:
        print("ERROR: No usable ASCII art found.")
        sys.exit(1)

    if args.auto_caption:
        todo = [i for i, c in enumerate(caption_ids) if c < 0]
        if todo:
            print(f"Auto-captioning {len(todo):,} untitled pieces...")
            generated = auto_caption_grids([grids[i] for i in todo])
            for i, caption in zip(todo, generated):
                if caption:
                    caption_ids[i] = caption_index.setdefault(
                        caption, len(caption_index))
            n_captioned = sum(1 for c in caption_ids if c >= 0)
            print(f"  Now {n_captioned:,}/{len(grids):,} captioned "
                  f"({len(caption_index):,} unique captions)")

    data = torch.stack(grids).to(torch.uint8)  # vocab fits in a byte
    payload = {"data": data}
    if caption_index:
        payload["caption_ids"] = torch.tensor(caption_ids, dtype=torch.long)
        payload["captions"] = list(caption_index)
    out_path = os.path.join(os.path.dirname(__file__), "human_data.pt")
    torch.save(payload, out_path)
    print(f"Saved {data.shape} to {out_path} "
          f"({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("\n=== Sample Human ASCII Art (normalized) ===\n")
    for i in range(min(3, len(grids))):
        print(f"--- Sample {i} ---")
        print(grid_to_string(grids[i]))
        print()


if __name__ == "__main__":
    main()
