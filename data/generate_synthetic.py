"""Synthetic captioned training data: prompt -> local t2i -> ASCII converter.

The data engine that fixes the model's real ceiling. ImageNet-Sketch gives
1,000 single-word class captions; this generates a corpus with arbitrary
short captions (attributes, counts, two-subject compositions) by rendering
prompts with a small local text-to-image model in a line-drawing style and
converting the images through the existing 6D shape-matching pipeline.
Output uses the standard {data, caption_ids, captions} payload, so
training consumes it unchanged:

    python data/generate_synthetic.py --num-samples 200000
    python training/train.py --init-from checkpoints/geometry_best.pt \
        --stages shading,human --shading-data data/synthetic_data.pt

Requires a GPU and: pip install diffusers accelerate
(--merge data/shading_data.pt appends the sketch dataset so both sources
train together.)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W
from data.charset import grid_to_string
from data.prompt_bank import build_prompts

# Style wrapper for the t2i model. The BARE caption is what gets stored —
# the model should learn "a dragon", not the rendering instructions.
STYLE = ("minimal black ink line drawing of {}, clean thick outlines, "
         "white background, no shading, centered")
NEGATIVE = "photo, color, shading, background clutter, text, watermark"

# Reject conversions that came out empty or muddy
MIN_INK_FRAC = 0.02
MAX_INK_FRAC = 0.55

DEFAULT_MODEL = "stabilityai/sd-turbo"


def _build_tables():
    """The converter's precomputed matcher tables (same recipe as
    generate_shading.generate_dataset)."""
    from data.generate_shading import _make_circle_masks, compute_shape_vectors
    masks = _make_circle_masks()
    mask_sums = np.array([m.sum() for m in masks], dtype=np.float32)
    shape_vectors, char_indices = compute_shape_vectors(masks)
    sv_sq_sum = (shape_vectors ** 2).sum(axis=1, keepdims=True).T
    return shape_vectors, masks, char_indices, mask_sums, sv_sq_sum


def _load_pipeline(model_id, device):
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as e:
        raise ImportError(
            "The data engine needs a text-to-image model: "
            "pip install diffusers accelerate") from e
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id,
                                                     torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def images_to_grids(pil_images, tables):
    """Convert PIL images -> list of [GRID_H, GRID_W] uint8 grids or None
    (None = rejected by the ink filter)."""
    from data.generate_shading import _pil_to_tensor, _to_gray_u8, \
        image_to_ascii_grid
    shape_vectors, masks, char_indices, mask_sums, sv_sq_sum = tables
    out = []
    total = GRID_H * GRID_W
    for img in pil_images:
        gray = _to_gray_u8(_pil_to_tensor(img))
        grid = image_to_ascii_grid(gray, shape_vectors, masks, char_indices,
                                   mask_sums, sv_sq_sum)
        ink = int((grid > 2).sum())
        if MIN_INK_FRAC * total <= ink <= MAX_INK_FRAC * total:
            out.append(grid.to(torch.uint8))
        else:
            out.append(None)
    return out


def _save(out_path, grids, caption_ids, captions, merge_payload=None):
    data = torch.stack(grids)
    ids = torch.tensor(caption_ids, dtype=torch.long)
    caps = list(captions)
    if merge_payload is not None:
        m_data = merge_payload["data"]
        m_ids = merge_payload.get("caption_ids")
        m_caps = list(merge_payload.get("captions") or [])
        if m_ids is None:
            m_ids = torch.full((len(m_data),), -1, dtype=torch.long)
        else:
            m_ids = m_ids.long().clone()
            m_ids[m_ids >= 0] += len(caps)
        data = torch.cat([data, m_data.to(torch.uint8)])
        ids = torch.cat([ids, m_ids])
        caps = caps + m_caps
    torch.save({"data": data, "caption_ids": ids, "captions": caps},
               out_path)
    return len(data)


def generate_dataset(num_samples, out_path, model_id=DEFAULT_MODEL,
                     batch_size=8, steps=2, seed=0, merge=None,
                     device=None, save_every=5_000, pipe=None):
    """Generate `num_samples` captioned grids and save the payload.

    pipe: injectable for tests — any callable matching the diffusers
    text2img interface (returns an object with .images).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if pipe is None:
        print(f"Loading {model_id} on {device} ...")
        pipe = _load_pipeline(model_id, device)

    tables = _build_tables()
    # Over-provision prompts: some images fail the ink filter
    prompts = build_prompts(int(num_samples * 1.5) + 64, seed=seed)
    print(f"Prompt bank: {len(prompts):,} unique captions "
          f"(e.g. {prompts[:3]})")

    merge_payload = None
    if merge:
        merge_payload = torch.load(merge, weights_only=True)
        if not isinstance(merge_payload, dict):
            merge_payload = {"data": merge_payload}
        print(f"Will merge {len(merge_payload['data']):,} samples "
              f"from {merge}")

    generator = torch.Generator(device=device.type).manual_seed(seed)
    grids, caption_ids = [], []
    caption_index = {}
    rejected = 0
    cursor = 0
    t0 = time.time()
    next_report, next_save = 0, save_every

    while len(grids) < num_samples and cursor < len(prompts):
        batch = prompts[cursor:cursor + batch_size]
        cursor += len(batch)
        result = pipe(prompt=[STYLE.format(p) for p in batch],
                      negative_prompt=[NEGATIVE] * len(batch),
                      num_inference_steps=steps, guidance_scale=0.0,
                      generator=generator)
        for caption, grid in zip(batch, images_to_grids(result.images,
                                                        tables)):
            if grid is None:
                rejected += 1
                continue
            grids.append(grid)
            caption_ids.append(caption_index.setdefault(caption,
                                                        len(caption_index)))

        done = len(grids)
        if done >= next_report:
            rate = done / (time.time() - t0)
            eta = (num_samples - done) / max(rate, 1e-9)
            print(f"  {done:,}/{num_samples:,} kept ({rejected:,} rejected)"
                  f"  {rate:.1f} img/s  ETA {eta/3600:.1f}h", flush=True)
            # First feedback after one batch, then every 500
            next_report = done + (500 if done else 1)
        if done >= next_save or done >= num_samples:
            n = _save(out_path, grids, caption_ids, list(caption_index),
                      merge_payload)
            print(f"  [checkpointed {n:,} samples to {out_path}]",
                  flush=True)
            next_save = done + save_every

    if not grids:
        print("ERROR: no usable images generated.")
        sys.exit(1)
    n = _save(out_path, grids, caption_ids, list(caption_index),
              merge_payload)
    print(f"\nSaved {n:,} samples, {len(caption_index):,} unique captions "
          f"to {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    print("\n=== Sample conversions ===")
    for i in range(min(2, len(grids))):
        cap = list(caption_index)[caption_ids[i]]
        print(f'--- "{cap}" ---')
        print(grid_to_string(grids[i]))
    return n


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-samples", type=int, default=200_000)
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "synthetic_data.pt"))
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="any diffusers text2img id "
                             f"(default: {DEFAULT_MODEL})")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2,
                        help="denoising steps (sd-turbo: 1-4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--merge", default=None,
                        help="existing dataset file to append (e.g. "
                             "data/shading_data.pt)")
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.out, model_id=args.model,
                     batch_size=args.batch, steps=args.steps,
                     seed=args.seed, merge=args.merge)


if __name__ == "__main__":
    main()
