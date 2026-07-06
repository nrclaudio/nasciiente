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

# Style wrapper for the t2i model. The BARE caption (plus a short style
# tag) is what gets stored — the model should learn "a dragon", not the
# rendering instructions. Icon phrasing won the preview tuning: bold
# isolated silhouettes survive 48x80 conversion; "line drawing" styles
# produced shading and thin strokes that converted to soup.
STYLE = ("simple flat black and white icon of {}, bold shapes, centered, "
         "isolated on plain white background")
NEGATIVE = "photo, color, shading, background clutter, text, watermark"

# Reject conversions that came out empty or muddy
MIN_INK_FRAC = 0.02
MAX_INK_FRAC = 0.55

# Three visual dialects, chosen per batch (--mix). v1 trained only on
# "filled": Otsu binarization reduced every image to a solid silhouette,
# so solid silhouettes are all the model could draw. The caption carries
# a style tag, so at inference the PROMPT selects the dialect ("a goat"
# / "a goat, outline style" / "a goat, shaded").
STYLE_MODES = {
    # v1 dialect: bold solid silhouettes. Tag-free so it matches the v1
    # dataset's plain captions when the two are merged.
    "filled": dict(style=STYLE, negative=NEGATIVE, binarize=True,
                   outline=False, max_ink=MAX_INK_FRAC,
                   flatten_bg=False, tag=""),
    # Boundary strokes. Derived from the SAME reliable icon prompt by
    # morphological edge extraction on the binary mask — prompting the
    # t2i for "line drawing" directly gave conversion soup in v1 tuning.
    "outline": dict(style=STYLE, negative=NEGATIVE, binarize=True,
                    outline=True, max_ink=0.35,
                    flatten_bg=False, tag=", outline style"),
    # Full grayscale through the converter's tonal pipeline (adaptive
    # gamma -> CLAHE -> Sobel blend -> 6D matching) — the density-ramp
    # look of classic ASCII art. Binarize OFF is the whole point.
    # flatten_bg is essential: CLAHE amplifies the near-white paper
    # texture of t2i outputs into faint glyphs across the whole canvas,
    # which sent 99% of tonal conversions over the ink cap in the first
    # v2 run (778 kept of ~78k attempted).
    "tonal": dict(style=("a detailed grayscale pencil drawing of {}, "
                         "soft shading, smooth gradients, centered, "
                         "isolated on a plain white background"),
                  negative=("color, photo, text, watermark, frame, "
                            "border, background texture, pattern"),
                  binarize=False, outline=False, max_ink=0.85,
                  flatten_bg=True, tag=", shaded"),
}
DEFAULT_MIX = "filled=0.20,outline=0.35,tonal=0.45"

DEFAULT_MODEL = "stabilityai/sd-turbo"


def parse_mix(spec):
    """'filled=0.2,tonal=0.8' -> normalized {mode: weight} dict."""
    mix = {}
    for part in spec.split(","):
        name, _, w = part.partition("=")
        name = name.strip()
        if name not in STYLE_MODES:
            raise ValueError(f"Unknown style mode {name!r} "
                             f"(choose from {sorted(STYLE_MODES)})")
        mix[name] = float(w)
    total = sum(mix.values())
    if total <= 0:
        raise ValueError("Style mix weights must sum to > 0")
    return {k: v / total for k, v in mix.items()}


def _build_tables():
    """The converter's precomputed matcher tables (same recipe as
    generate_shading.generate_dataset)."""
    from data.generate_shading import _make_circle_masks, compute_shape_vectors
    masks = _make_circle_masks()
    mask_sums = np.array([m.sum() for m in masks], dtype=np.float32)
    shape_vectors, char_indices = compute_shape_vectors(masks)
    sv_sq_sum = (shape_vectors ** 2).sum(axis=1, keepdims=True).T
    return shape_vectors, masks, char_indices, mask_sums, sv_sq_sum


def _load_pipeline(model_id, device, cpu_offload=None):
    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError as e:
        raise ImportError(
            "The data engine needs a text-to-image model: "
            "pip install diffusers accelerate") from e
    is_flux = "flux" in model_id.lower()
    if device.type == "cuda":
        # FLUX overflows in fp16; bf16 is its native training dtype
        dtype = torch.bfloat16 if is_flux else torch.float16
    else:
        dtype = torch.float32
    pipe = AutoPipelineForText2Image.from_pretrained(model_id,
                                                     torch_dtype=dtype)
    if cpu_offload is None:
        # FLUX.1-schnell (12B) + its T5 encoder need ~34GB — more than
        # most single consumer GPUs. Offload keeps only the active
        # component on the GPU at some throughput cost.
        cpu_offload = is_flux and device.type == "cuda"
    if cpu_offload:
        pipe.enable_model_cpu_offload()
        print("Pipeline: CPU offload enabled (large model)")
    else:
        pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _pipe_kwarg_filter(pipe):
    """Not every pipeline takes every argument (FLUX has no
    negative_prompt — it is CFG-free by construction). Filter the call
    kwargs down to what this pipeline's signature actually accepts."""
    import inspect
    try:
        params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        return lambda kw: kw
    if any(p.kind == inspect.Parameter.VAR_KEYWORD
           for p in params.values()):
        return lambda kw: kw
    return lambda kw: {k: v for k, v in kw.items() if k in params}


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


def _flatten_background(gray_u8, margin=25):
    """Clamp near-background pixels to pure white.

    The background level is estimated from the image border (the same
    trick as the converter's auto-polarity). Everything within `margin`
    gray levels of it becomes exactly 255, so paper texture, vignettes
    and JPEG shimmer convert to space instead of a canvas-wide film of
    faint glyphs once CLAHE has amplified them. Dark-background images
    (not expected from our prompts) pass through untouched.
    """
    border = torch.cat([gray_u8[0], gray_u8[-1],
                        gray_u8[:, 0], gray_u8[:, -1]]).float()
    bg = border.median()
    if bg <= 128:
        return gray_u8
    out = gray_u8.clone()
    out[gray_u8.float() >= bg - margin] = 255
    return out


def _mask_outline(bin_u8, thickness=3):
    """Reduce a filled binary shape (0=ink, 255=paper) to its boundary.

    Morphological edge: ink minus its erosion leaves a ring `thickness`
    pixels wide. Turns the engine's solid silhouettes into stroke art —
    a second visual dialect from the same generated images.
    """
    ink = (bin_u8 == 0).float()[None, None]
    k = 2 * thickness + 1
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - ink, k, stride=1,
                                                  padding=thickness)
    ring = (ink - eroded).squeeze() > 0.5
    return torch.where(ring, torch.zeros_like(bin_u8),
                       torch.full_like(bin_u8, 255))


def images_to_grids(pil_images, tables, min_ink=MIN_INK_FRAC,
                    max_ink=MAX_INK_FRAC, binarize=False, trim=0.04,
                    outline=False, flatten_bg=False):
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
        if flatten_bg:
            gray = _flatten_background(gray)
        if binarize:
            gray = _binarize(gray)
        if outline:
            gray = _mask_outline(gray)
        grid = image_to_ascii_grid(gray, shape_vectors, masks, char_indices,
                                   mask_sums, sv_sq_sum)
        ink_frac = float((grid > 2).sum()) / total
        if min_ink <= ink_frac <= max_ink:
            out.append((grid.to(torch.uint8), ink_frac))
        else:
            out.append((None, ink_frac))
    return out


# ---- worker-side conversion (multiprocessing) ---------------------------
# The ASCII conversion is pure CPU and was serializing the GPU: the
# pipeline generated a batch, then idled while one core converted it.
# A pool converts batch N while the GPU generates batch N+1.
_WORKER = {}


def _worker_init(trim):
    import contextlib
    import io

    import torch as _torch
    _torch.set_num_threads(1)  # one image per task; don't thrash cores
    with contextlib.redirect_stdout(io.StringIO()):  # table-build prints
        _WORKER["tables"] = _build_tables()
    _WORKER["trim"] = trim


def _worker_convert(job):
    # Conversion params travel with each image because style modes vary
    # per batch (tonal converts un-binarized, outline post-processes...)
    image, min_ink, max_ink, binarize, outline, flatten_bg = job
    grid, ink_frac = images_to_grids([image], _WORKER["tables"],
                                     min_ink=min_ink, max_ink=max_ink,
                                     binarize=binarize, outline=outline,
                                     flatten_bg=flatten_bg,
                                     trim=_WORKER["trim"])[0]
    # Return numpy, NOT a torch tensor: torch pickles tensors across
    # processes via fd-sharing, which exhausts descriptors under a steady
    # stream of results ("received 0 items of ancdata"). A 48x80 uint8
    # array is 3.8KB of plain bytes.
    return (None if grid is None else grid.numpy(), ink_frac)


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
                     trim=0.04, clip_filter=0.0, clip_scorer=None,
                     workers=None, modes=None, cpu_offload=None):
    """Generate `num_samples` captioned grids and save the payload.

    pipe: injectable for tests — any callable matching the diffusers
    text2img interface (returns an object with .images).
    preview_dir: also dump every processed sample there as an image/grid
    pair named with its caption, ink fraction and kept/rejected verdict —
    run with --num-samples 24 --preview-dir ... to eyeball what the
    filter is doing before a long run.
    clip_filter: when > 0, reject images whose CLIP similarity to their
    own bare caption falls below the threshold (subject misses, fused
    chimeras scoring low, degenerate outputs). clip_scorer is injectable
    for tests: callable(pil_images, captions) -> scores tensor.
    workers: conversion worker processes. None = cpu_count-2; <=1 =
    serial. Preview mode forces serial (it pairs images with grids).
    modes: None (legacy single-style path using `style`/`binarize`) or a
    {mode_name: weight} dict over STYLE_MODES — each batch draws a mode,
    renders with that mode's t2i style, converts with its settings, and
    stores captions with its style tag ("a goat, shaded").
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    if pipe is None:
        print(f"Loading {model_id} on {device} ...")
        pipe = _load_pipeline(model_id, device, cpu_offload=cpu_offload)
    kwarg_filter = _pipe_kwarg_filter(pipe)

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

    if clip_filter > 0 and clip_scorer is None:
        from data.clip_rank import clip_image_scores

        def clip_scorer(images, captions):
            return clip_image_scores(images, captions,
                                     device=device.type)

    # Per-batch style modes: legacy args become a single always-on mode
    if modes:
        mix = ({k: v for k, v in modes.items()} if isinstance(modes, dict)
               else parse_mix(modes))
        total = sum(mix.values())
        mix = {k: v / total for k, v in mix.items()}
        print("Style mix: " + ", ".join(f"{k}={v:.0%}"
                                        for k, v in mix.items()))
    else:
        mix = {None: 1.0}
    legacy_mode = dict(style=style, negative=NEGATIVE, binarize=binarize,
                       outline=False, max_ink=max_ink, flatten_bg=False,
                       tag="")
    import random as _random
    mode_rng = _random.Random(seed ^ 0x5F17)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    if preview_dir:
        workers = 1
    pool = None
    max_pending = 3  # batches in flight; bounds RAM and keeps order fresh
    if workers > 1:
        import multiprocessing as mp
        # spawn, not fork: the parent holds a CUDA context
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers, initializer=_worker_init,
                        initargs=(trim,))
        print(f"Conversion pool: {workers} workers "
              f"(GPU generates while CPUs convert)")

    generator = torch.Generator(device=device.type).manual_seed(seed)
    grids, caption_ids = [], []
    caption_index = {}
    too_blank = too_dense = off_prompt = 0
    processed = 0
    cursor = 0
    t0 = time.time()
    next_report, next_save = 0, save_every

    def absorb(captions_batch, images, converted):
        nonlocal processed, too_blank, too_dense, next_report, next_save
        for caption, image, (grid, ink_frac) in zip(captions_batch, images,
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
                  f"{too_dense:,} too dense, {off_prompt:,} off-prompt)"
                  f"  {rate:.1f} img/s  ETA {eta/3600:.1f}h", flush=True)
            # First feedback after one batch, then every 500
            next_report = done + (500 if done else 1)
        if done and done >= next_save:
            n = _save(out_path, grids, caption_ids, list(caption_index),
                      merge_payload)
            print(f"  [checkpointed {n:,} samples to {out_path}]",
                  flush=True)
            next_save = done + save_every

    from collections import deque
    pending = deque()

    def drain(block=False):
        while pending and (block or len(pending) > max_pending
                           or pending[0][0].ready()):
            res, caps, imgs = pending.popleft()
            converted = [(None if g is None else torch.from_numpy(g), ink)
                         for g, ink in res.get()]
            absorb(caps, imgs, converted)

    while True:
        # Count in-flight images as kept (conservative) so the pool path
        # doesn't over-generate past num_samples while conversions drain
        in_flight = sum(len(im) for _, _, im in pending)
        if len(grids) + in_flight >= num_samples or cursor >= len(prompts):
            drain(block=True)
            if len(grids) >= num_samples or cursor >= len(prompts):
                break
            continue
        batch = prompts[cursor:cursor + batch_size]
        cursor += len(batch)
        mode_name = mode_rng.choices(list(mix),
                                     weights=list(mix.values()))[0]
        m = STYLE_MODES[mode_name] if mode_name else legacy_mode
        result = pipe(**kwarg_filter(dict(
            prompt=[m["style"].format(p) for p in batch],
            negative_prompt=[m["negative"]] * len(batch),
            num_inference_steps=steps, guidance_scale=0.0,
            height=512, width=512, generator=generator)))
        images = list(result.images)
        captions_batch = list(batch)   # bare captions: what CLIP scores
        if clip_filter > 0:
            sims = clip_scorer(images, captions_batch)
            keep = [float(s) >= clip_filter for s in sims]
            off_prompt += keep.count(False)
            images = [im for im, k in zip(images, keep) if k]
            captions_batch = [c for c, k in zip(captions_batch, keep) if k]
            if not images:
                continue
        # Stored captions carry the style tag, so at inference the
        # prompt selects the dialect ("a goat, shaded")
        stored = [c + m["tag"] for c in captions_batch]
        if pool is not None:
            jobs = [(im, min_ink, m["max_ink"], m["binarize"],
                     m["outline"], m["flatten_bg"]) for im in images]
            pending.append((pool.map_async(_worker_convert, jobs),
                            stored, images))
            drain()
        else:
            absorb(stored, images,
                   images_to_grids(images, tables, min_ink=min_ink,
                                   max_ink=m["max_ink"],
                                   binarize=m["binarize"],
                                   outline=m["outline"],
                                   flatten_bg=m["flatten_bg"], trim=trim))

    drain(block=True)
    if pool is not None:
        pool.close()
        pool.join()

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
    parser.add_argument("--steps", type=int, default=4,
                        help="denoising steps (sd-turbo: 1-4; 4 won the "
                             "preview tuning)")
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
    parser.add_argument("--workers", type=int, default=None,
                        help="conversion worker processes (default: "
                             "cpu_count-2; 1 = serial)")
    parser.add_argument("--clip-filter", type=float, default=0.2,
                        help="reject images whose CLIP similarity to their "
                             "own caption is below this (0 = off)")
    parser.add_argument("--trim", type=float, default=0.04,
                        help="fraction cropped off every image edge before "
                             "conversion (kills t2i border bands)")
    parser.add_argument("--binarize", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="threshold images to pure black/white before "
                             "conversion — strips the shading t2i models "
                             "sneak in despite line-drawing prompts "
                             "(--no-binarize to disable; ignored when "
                             "--mix is given)")
    parser.add_argument("--mix", default=None,
                        help="style-mode mix, e.g. "
                             f"'{DEFAULT_MIX}' — batches draw a mode "
                             f"from {sorted(STYLE_MODES)} and captions "
                             "carry the mode's style tag. Omit for the "
                             "legacy single-style path.")
    parser.add_argument("--cpu-offload",
                        action=argparse.BooleanOptionalAction,
                        default=None,
                        help="offload pipeline components to CPU RAM "
                             "(default: auto — on for FLUX-size models, "
                             "off otherwise)")
    args = parser.parse_args()
    generate_dataset(args.num_samples, args.out, model_id=args.model,
                     batch_size=args.batch, steps=args.steps,
                     seed=args.seed, merge=args.merge,
                     min_ink=args.min_ink, max_ink=args.max_ink,
                     style=args.style, preview_dir=args.preview_dir,
                     binarize=args.binarize, trim=args.trim,
                     clip_filter=args.clip_filter, workers=args.workers,
                     modes=parse_mix(args.mix) if args.mix else None,
                     cpu_offload=args.cpu_offload)


if __name__ == "__main__":
    main()
