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


def _binarize(gray_u8):
    """Force a shaded image into two-tone line art via Otsu's threshold.

    Finds the split that maximizes between-class variance — the natural
    dark/light boundary of THIS image — instead of forcing a fixed ink
    quota (a fixed percentile fattens strokes and drags shading along on
    mostly-white images). Genuinely unimodal images threshold to near
    nothing and get caught by the too-blank filter downstream.
    """
    hist = torch.bincount(gray_u8.flatten().long(), minlength=256).float()
    total = hist.sum()
    omega = torch.cumsum(hist, 0) / total                    # P(class 0)
    mu = torch.cumsum(hist * torch.arange(256), 0) / total   # cum. mean
    mu_t = mu[-1]
    between = (mu_t * omega - mu).pow(2) / (omega * (1 - omega)).clamp_min(
        1e-9)
    thresh = int(between.argmax())
    return torch.where(gray_u8.long() <= thresh,
                       torch.zeros_like(gray_u8),
                       torch.full_like(gray_u8, 255)).to(torch.uint8)


def images_to_grids(pil_images, tables, min_ink=MIN_INK_FRAC,
                    max_ink=MAX_INK_FRAC, binarize=False, trim=0.04):
    """Convert PIL images -> list of (grid_or_None, ink_fraction).

    grid is a [GRID_H, GRID_W] uint8 tensor, or None when the conversion
    fell outside the [min_ink, max_ink] band (the ink fraction is still
    reported so callers can diagnose WHY images are being rejected).
    trim crops that fraction off every image edge first — t2i outputs
    often carry border bands/vignettes that would otherwise train the
    model to draw sidebars.
    """
    from data.generate_shading import _pil_to_tensor, _to_gray_u8, \
        image_to_ascii_grid
    shape_vectors, masks, char_indices, mask_sums, sv_sq_sum = tables
    out = []
    total = GRID_H * GRID_W
    for img in pil_images:
        gray = _to_gray_u8(_pil_to_tensor(img))
        if trim > 0:
            h, w = gray.shape
            dh, dw = int(h * trim), int(w * trim)
            gray = gray[dh:h - dh, dw:w - dw]
        if binarize:
            gray = _binarize(gray)
        grid = image_to_ascii_grid(gray, shape_vectors, masks, char_indices,
                                   mask_sums, sv_sq_sum)
        ink_frac = float((grid > 2).sum()) / total
        if min_ink <= ink_frac <= max_ink:
            out.append((grid.to(torch.uint8), ink_frac))
        else:
            out.append((None, ink_frac))
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
                     device=None, save_every=5_000, pipe=None,
                     min_ink=MIN_INK_FRAC, max_ink=MAX_INK_FRAC,
                     style=STYLE, preview_dir=None, binarize=False,
                     trim=0.04):
    """Generate `num_samples` captioned grids and save the payload.

    pipe: injectable for tests — any callable matching the diffusers
    text2img interface (returns an object with .images).
    preview_dir: also dump every processed sample there as an image/grid
    pair named with its caption, ink fraction and kept/rejected verdict —
    run with --num-samples 24 --preview-dir ... to eyeball what the
    filter is doing before a long run.
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
    prompts = build_prompts(int(num_samples * 3) + 64, seed=seed)
    print(f"Prompt bank: {len(prompts):,} captions "
          f"(e.g. {prompts[:3]})")
    if preview_dir:
        os.makedirs(preview_dir, exist_ok=True)

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
    too_blank = too_dense = 0
    processed = 0
    cursor = 0
    t0 = time.time()
    next_report, next_save = 0, save_every

    while len(grids) < num_samples and cursor < len(prompts):
        batch = prompts[cursor:cursor + batch_size]
        cursor += len(batch)
        result = pipe(prompt=[style.format(p) for p in batch],
                      negative_prompt=[NEGATIVE] * len(batch),
                      num_inference_steps=steps, guidance_scale=0.0,
                      generator=generator)
        converted = images_to_grids(result.images, tables,
                                    min_ink=min_ink, max_ink=max_ink,
                                    binarize=binarize, trim=trim)
        for caption, image, (grid, ink_frac) in zip(batch, result.images,
                                                    converted):
            processed += 1
            if preview_dir:
                verdict = "kept" if grid is not None else "rejected"
                stem = (f"{processed:03d}_{verdict}_ink{ink_frac:.3f}_"
                        + "".join(c if c.isalnum() else "_"
                                  for c in caption)[:40])
                image.save(os.path.join(preview_dir, stem + ".png"))
                if grid is not None:
                    with open(os.path.join(preview_dir, stem + ".txt"),
                              "w") as f:
                        f.write(grid_to_string(grid.long()))
            if grid is None:
                if ink_frac < min_ink:
                    too_blank += 1
                else:
                    too_dense += 1
                continue
            grids.append(grid)
            caption_ids.append(caption_index.setdefault(caption,
                                                        len(caption_index)))

        done = len(grids)
        if done >= next_report or processed <= batch_size:
            rate = done / (time.time() - t0)
            eta = (num_samples - done) / max(rate, 1e-9)
            print(f"  {done:,}/{num_samples:,} kept "
                  f"(rejected: {too_blank:,} too blank, "
                  f"{too_dense:,} too dense)"
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
    parser.add_argument("--min-ink", type=float, default=MIN_INK_FRAC)
    parser.add_argument("--max-ink", type=float, default=MAX_INK_FRAC)
    parser.add_argument("--style", default=STYLE,
                        help="t2i style wrapper; must contain {} for the "
                             "caption")
    parser.add_argument("--preview-dir", default=None,
                        help="dump every image/grid pair here (use with a "
                             "small --num-samples to tune style/filters)")
    parser.add_argument("--trim", type=float, default=0.04,
                        help="fraction cropped off every image edge before "
                             "conversion (kills t2i border bands)")
    parser.add_argument("--binarize", action="store_true",
                        help="threshold images to pure black/white before "
                             "conversion — strips the shading t2i models "
                             "sneak in despite line-drawing prompts")
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.out, model_id=args.model,
                     batch_size=args.batch, steps=args.steps,
                     seed=args.seed, merge=args.merge,
                     min_ink=args.min_ink, max_ink=args.max_ink,
                     style=args.style, preview_dir=args.preview_dir,
                     binarize=args.binarize, trim=args.trim)


if __name__ == "__main__":
    main()
