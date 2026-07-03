"""Generate ASCII shading data from image datasets.

Supports multiple image sources: ImageNet-Sketch (default), ImageNet,
Imagenette, Caltech-256, Caltech-101, and STL-10 (legacy). Use --source
to select.

Uses 6D shape vector matching (Alex Harri, 2025): each cell is divided into
6 sub-regions (3 columns x 2 rows of sampling circles). Characters are matched
by Euclidean distance in this 6D space, with global and directional contrast
enhancement for edge clarity.

Pre-processing pipeline: Adaptive Gamma -> CLAHE -> Sobel edge blend.

All 95 printable ASCII characters are used.
Produces [N, 48, 80] long tensors.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W, SHADING_NUM_SAMPLES
from data.charset import char_to_idx, grid_to_string

# Sentinel for "no class label" in the aligned label lists; saved files use
# caption_id -1 the same way (see the {'data','caption_ids','captions'}
# payload in generate_dataset)
NO_LABEL = -1

# Sub-cell resolution for shape matching
CELL_H = 12
CELL_W = 8
NUM_REGIONS = 6  # 3 columns x 2 rows of sampling circles

# All printable ASCII characters (space=32 through ~=126)
PRINTABLE = [chr(c) for c in range(32, 127)]
NUM_CHARS = len(PRINTABLE)

# Contrast enhancement exponent (higher = more shape emphasis)
CONTRAST_POWER = 2.0
# Directional contrast boost factor
EDGE_BOOST = 2.0
# Sobel edge blend weight (blends edge map into brightness before matching)
EDGE_BLEND = 0.3

# Adaptive gamma: per-region gamma based on local brightness
AG_KERNEL = 48

# CLAHE: Contrast Limited Adaptive Histogram Equalization
CLAHE_TILE_H = 8
CLAHE_TILE_W = 10
CLAHE_CLIP = 3.0

# Minimum image dimension — skip images smaller than this
MIN_IMAGE_DIM = 64

# Ink below this (before contrast enhancement) renders as whitespace.
# Kills the "backtick halo" that adaptive gamma + CLAHE otherwise amplify
# out of near-flat backgrounds.
WHITESPACE_FLOOR = 0.06


def _find_mono_font(size=32):
    """Find a monospace font, falling back to PIL default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size), path
    return ImageFont.load_default(), "PIL default"


def _make_circle_masks():
    """Create 6 circle masks in a 3-col x 2-row arrangement.

    Returns list of 6 boolean arrays of shape [CELL_H, CELL_W].
    """
    radius = CELL_W / 3
    col_centers = [CELL_W / 6, CELL_W / 2, 5 * CELL_W / 6]
    row_centers = [CELL_H / 4, 3 * CELL_H / 4]

    yy, xx = np.mgrid[:CELL_H, :CELL_W].astype(np.float32)
    yy += 0.5  # pixel centers
    xx += 0.5

    masks = []
    for ry in row_centers:
        for cx in col_centers:
            dist = np.sqrt((xx - cx) ** 2 + (yy - ry) ** 2)
            masks.append(dist <= radius)

    return masks


def compute_shape_vectors(masks):
    """Compute 6D shape vectors for all 95 printable ASCII characters.

    Returns:
        shape_vectors: [NUM_CHARS, NUM_REGIONS] float32, normalized per dimension
        char_indices: [NUM_CHARS] int64, vocab indices for each character
    """
    font, font_name = _find_mono_font(size=32)
    print(f"  Font: {font_name}")

    bbox = font.getbbox("M")
    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    print(f"  Glyph render size: {fw}x{fh} -> downscaled to {CELL_W}x{CELL_H}")

    shape_vectors = np.zeros((NUM_CHARS, NUM_REGIONS), dtype=np.float32)

    for i, ch in enumerate(PRINTABLE):
        img = Image.new("L", (fw, fh), 0)
        draw = ImageDraw.Draw(img)
        draw.text((-bbox[0], -bbox[1]), ch, fill=255, font=font)
        img = img.resize((CELL_W, CELL_H), Image.LANCZOS)
        bitmap = np.array(img, dtype=np.float32) / 255.0

        for j, mask in enumerate(masks):
            if mask.any():
                shape_vectors[i, j] = bitmap[mask].mean()

    # Normalize: divide each dimension by its max across all characters
    max_per_dim = shape_vectors.max(axis=0, keepdims=True)
    max_per_dim = np.maximum(max_per_dim, 1e-8)
    shape_vectors = shape_vectors / max_per_dim

    char_indices = np.array([char_to_idx(ch) for ch in PRINTABLE], dtype=np.int64)

    # Print density ramp (sorted by total density for reference)
    total_density = shape_vectors.sum(axis=1)
    density_order = np.argsort(total_density)
    ramp = "".join(PRINTABLE[i] for i in density_order)
    print(f"  Shape ramp ({NUM_CHARS} chars): {ramp}")
    print(f"  Shape vector range: [{shape_vectors.min():.3f}, {shape_vectors.max():.3f}]")

    return shape_vectors, char_indices


def _adaptive_gamma(img_np):
    """Per-region adaptive gamma based on local brightness.

    Dark areas get gamma < 1 (brighten), bright areas get gamma > 1 (darken).
    This spreads the histogram locally for better character variety.
    """
    t = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0)
    k = AG_KERNEL
    weight = torch.ones(1, 1, k, k, dtype=torch.float32) / (k * k)
    pad = k // 2
    padded = F.pad(t, (pad, pad, pad, pad), mode="reflect")
    local_mean = F.conv2d(padded, weight)
    local_mean = local_mean[:, :, :t.shape[2], :t.shape[3]].squeeze().numpy()
    gamma_map = 0.4 + 1.2 * local_mean  # range ~[0.4, 1.6]
    return np.power(np.clip(img_np, 1e-6, 1.0), gamma_map).astype(np.float32)


def _clahe(img_np):
    """Contrast Limited Adaptive Histogram Equalization.

    Divides image into tiles, equalizes each with clipping to prevent
    noise amplification, then bilinear interpolates between tile mappings.
    """
    h, w = img_np.shape
    nbins = 256
    th = h // CLAHE_TILE_H
    tw = w // CLAHE_TILE_W

    # Build per-tile CDF mappings
    mappings = np.zeros((CLAHE_TILE_H, CLAHE_TILE_W, nbins), dtype=np.float32)
    for ti in range(CLAHE_TILE_H):
        for tj in range(CLAHE_TILE_W):
            r0, r1 = ti * th, min((ti + 1) * th, h)
            c0, c1 = tj * tw, min((tj + 1) * tw, w)
            tile = img_np[r0:r1, c0:c1].flatten()

            hist, _ = np.histogram(tile, bins=nbins, range=(0, 1))
            clip_val = max(1, int(CLAHE_CLIP * len(tile) / nbins))
            excess = np.maximum(hist - clip_val, 0).sum()
            hist = np.minimum(hist, clip_val)
            hist += excess // nbins

            cdf = hist.cumsum().astype(np.float64)
            cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
            denom = cdf[-1] - cdf_min
            if denom > 0:
                mappings[ti, tj] = ((cdf - cdf_min) / denom).astype(np.float32)
            else:
                mappings[ti, tj] = np.linspace(0, 1, nbins)

    # Vectorized bilinear interpolation between tile centers
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    ty = np.clip(yy / th - 0.5, 0, CLAHE_TILE_H - 1)
    tx = np.clip(xx / tw - 0.5, 0, CLAHE_TILE_W - 1)
    ti0 = np.floor(ty).astype(int)
    tj0 = np.floor(tx).astype(int)
    ti1 = np.minimum(ti0 + 1, CLAHE_TILE_H - 1)
    tj1 = np.minimum(tj0 + 1, CLAHE_TILE_W - 1)
    fy = ty - ti0
    fx = tx - tj0
    bin_idx = np.clip((img_np * nbins).astype(int), 0, nbins - 1)

    v00 = mappings[ti0, tj0, bin_idx]
    v01 = mappings[ti0, tj1, bin_idx]
    v10 = mappings[ti1, tj0, bin_idx]
    v11 = mappings[ti1, tj1, bin_idx]

    result = ((1 - fy) * (1 - fx) * v00 + (1 - fy) * fx * v01 +
              fy * (1 - fx) * v10 + fy * fx * v11)
    # Bins below a tile's first non-zero CDF entry map slightly negative
    # when a neighboring tile's mapping is interpolated in — clip them.
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def _border_ink_mean(ink):
    """Mean ink along the image border (~5% of the short side wide)."""
    h, w = ink.shape
    b = max(1, int(0.05 * min(h, w)))
    border = torch.cat([
        ink[:b].flatten(), ink[-b:].flatten(),
        ink[:, :b].flatten(), ink[:, -b:].flatten(),
    ])
    return border.mean().item()


def image_to_ascii_grid(img_tensor, shape_vectors, masks, char_indices,
                        mask_sums, sv_sq_sum, letterbox=True,
                        auto_polarity=True, whitespace_floor=WHITESPACE_FLOOR):
    """Convert a [3, H, W] image tensor to [GRID_H, GRID_W] char indices.

    Pipeline: grayscale -> invert -> upscale -> adaptive gamma -> CLAHE ->
              Sobel edge blend -> 6D shape matching -> contrast enhancement ->
              8-neighbor directional boost -> nearest character.

    Args:
        img_tensor: [3, H, W] float tensor in [0, 1]
        shape_vectors: [NUM_CHARS, NUM_REGIONS] normalized shape vectors
        masks: list of 6 boolean arrays [CELL_H, CELL_W]
        char_indices: [NUM_CHARS] vocab indices
        mask_sums: [NUM_REGIONS] precomputed sum of each mask
        sv_sq_sum: [1, NUM_CHARS] precomputed squared norms of shape vectors
        letterbox: preserve aspect ratio, padding with whitespace
        auto_polarity: flip ink so the (border-estimated) background is sparse
        whitespace_floor: ink below this renders as space (0 disables)

    Also accepts the compact loader format: [H, W] (or [1, H, W]) uint8
    grayscale, as produced by _to_gray_u8.
    """
    # Grayscale -> invert -> [0, 1]
    if img_tensor.dtype == torch.uint8:
        gray = img_tensor.squeeze(0) if img_tensor.ndim == 3 else img_tensor
        gray = gray.float() / 255.0
    else:
        gray = (0.2989 * img_tensor[0] + 0.5870 * img_tensor[1]
                + 0.1140 * img_tensor[2])
    ink = 1.0 - gray

    # Flip polarity when the background (border) would render dense —
    # dark photos otherwise become walls of 'N'/'M'.
    if auto_polarity and _border_ink_mean(ink) > 0.5:
        ink = 1.0 - ink

    # Upscale to sub-cell resolution
    inv_4d = ink.unsqueeze(0).unsqueeze(0)
    target_h, target_w = GRID_H * CELL_H, GRID_W * CELL_W
    if letterbox:
        h, w = ink.shape
        scale = min(target_h / h, target_w / w)
        nh = max(1, min(target_h, round(h * scale)))
        nw = max(1, min(target_w, round(w * scale)))
        resized = F.interpolate(inv_4d, size=(nh, nw),
                                mode="bilinear", align_corners=False)
        canvas = torch.zeros(1, 1, target_h, target_w)
        top, left = (target_h - nh) // 2, (target_w - nw) // 2
        canvas[:, :, top:top + nh, left:left + nw] = resized
        upscaled = canvas.squeeze().numpy()
    else:
        upscaled = F.interpolate(
            inv_4d, size=(target_h, target_w),
            mode="bilinear", align_corners=False
        ).squeeze().numpy()

    # Remember pre-enhancement ink: pixels that start below the whitespace
    # floor must stay blank no matter what the contrast chain does to them.
    raw_ink = upscaled.copy()

    # --- Adaptive gamma ---
    upscaled = _adaptive_gamma(upscaled)

    # --- CLAHE ---
    upscaled = _clahe(upscaled)

    # --- Sobel edge blending ---
    up_t = torch.from_numpy(upscaled).float()
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
    g = up_t.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(g, kx, padding=1).squeeze()
    gy = F.conv2d(g, ky, padding=1).squeeze()
    edge_map = torch.sqrt(gx ** 2 + gy ** 2)
    edge_max = edge_map.max()
    if edge_max > 0:
        edge_map = edge_map / edge_max
    upscaled = torch.clamp(up_t + EDGE_BLEND * edge_map, 0, 1).numpy()

    # --- Whitespace floor ---
    if whitespace_floor > 0:
        upscaled[raw_ink < whitespace_floor] = 0.0

    # Reshape into cells: [GRID_H, GRID_W, CELL_H, CELL_W]
    cells = (upscaled.reshape(GRID_H, CELL_H, GRID_W, CELL_W)
             .transpose(0, 2, 1, 3))

    # Compute 6D sampling vectors for all cells
    sampling = np.zeros((GRID_H, GRID_W, NUM_REGIONS), dtype=np.float32)
    for k, mask in enumerate(masks):
        sampling[:, :, k] = (
            (cells * mask[np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3))
            / mask_sums[k]
        )

    # --- Global contrast enhancement ---
    max_vals = sampling.max(axis=2, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-8)
    normed = sampling / max_vals
    enhanced = (normed ** CONTRAST_POWER) * max_vals

    # --- Directional contrast enhancement ---
    # Compare each cell with its 8 neighbors; boost where edges are detected
    padded = np.pad(enhanced, ((1, 1), (1, 1), (0, 0)), mode="edge")
    neighbors = np.stack([
        padded[:-2, 1:-1],   # up
        padded[2:, 1:-1],    # down
        padded[1:-1, :-2],   # left
        padded[1:-1, 2:],    # right
        padded[:-2, :-2],    # up-left
        padded[:-2, 2:],     # up-right
        padded[2:, :-2],     # down-left
        padded[2:, 2:],      # down-right
    ])
    max_diff = np.abs(neighbors - enhanced[np.newaxis]).max(axis=0)
    boost = 1.0 + EDGE_BOOST * max_diff
    enhanced = enhanced * boost

    # --- Find nearest character via Euclidean distance in 6D ---
    flat = enhanced.reshape(-1, NUM_REGIONS)  # [GRID_H*GRID_W, 6]

    # Efficient distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    flat_sq_sum = (flat ** 2).sum(axis=1, keepdims=True)  # [N, 1]
    dists = flat_sq_sum + sv_sq_sum - 2.0 * (flat @ shape_vectors.T)  # [N, C]

    best = np.argmin(dists, axis=1)
    result = char_indices[best].reshape(GRID_H, GRID_W)

    return torch.from_numpy(result)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _pil_to_tensor(pil_img):
    """Convert a PIL image to a [3, H, W] float tensor in [0, 1]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return T.ToTensor()(pil_img)


def _to_gray_u8(img_tensor):
    """Compress a [3, H, W] float image for in-RAM storage.

    Grayscale uint8, downscaled to fit the conversion canvas. Holding
    tens of thousands of full-size RGB float32 images needs hundreds of
    GB of RAM; this is ~40x smaller with zero pipeline quality loss —
    the converter grayscales and letterboxes into the same canvas anyway.

    Returns [H', W'] uint8 with max dims (GRID_H*CELL_H, GRID_W*CELL_W).
    """
    gray = (0.2989 * img_tensor[0] + 0.5870 * img_tensor[1]
            + 0.1140 * img_tensor[2])
    h, w = gray.shape
    target_h, target_w = GRID_H * CELL_H, GRID_W * CELL_W
    scale = min(target_h / h, target_w / w)
    if scale < 1.0:  # only ever downscale; upscaling happens at convert time
        nh = max(1, round(h * scale))
        nw = max(1, round(w * scale))
        gray = F.interpolate(gray[None, None], size=(nh, nw),
                             mode="area").squeeze(0).squeeze(0)
    return (gray.clamp(0, 1) * 255).to(torch.uint8)


def _load_stl10(data_dir):
    """Load STL-10 images (legacy). Returns list of [3, H, W] tensors."""
    print("\nLoading STL-10...")
    transform = T.ToTensor()
    all_images = []
    for split in ["unlabeled", "train", "test"]:
        ds = torchvision.datasets.STL10(
            root=data_dir, split=split, download=True, transform=transform)
        for img, _ in ds:
            all_images.append(_to_gray_u8(img))
        print(f"  {split}: {len(ds):,} images ({all_images[-1].shape})")
    print(f"Total STL-10 images: {len(all_images):,}")
    return all_images


def _load_imagenette(data_dir):
    """Load Imagenette (320px) via torchvision. Returns list of tensors."""
    print("\nLoading Imagenette (320px)...")
    all_images = []
    for split in ["train", "val"]:
        ds = torchvision.datasets.Imagenette(
            root=data_dir, split=split, size="320px", download=True)
        for img, _ in ds:
            tensor = _pil_to_tensor(img)
            if min(tensor.shape[1], tensor.shape[2]) >= MIN_IMAGE_DIM:
                all_images.append(_to_gray_u8(tensor))
        print(f"  {split}: {len(ds):,} images")
    print(f"Total Imagenette images: {len(all_images):,}")
    return all_images


def _load_caltech101(data_dir):
    """Load Caltech-101 via torchvision. Returns list of tensors."""
    print("\nLoading Caltech-101...")
    ds = torchvision.datasets.Caltech101(root=data_dir, download=True)
    all_images = []
    skipped = 0
    for i in range(len(ds)):
        img, _ = ds[i]
        tensor = _pil_to_tensor(img)
        if min(tensor.shape[1], tensor.shape[2]) >= MIN_IMAGE_DIM:
            all_images.append(_to_gray_u8(tensor))
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} images smaller than {MIN_IMAGE_DIM}px")
    print(f"Total Caltech-101 images: {len(all_images):,}")
    return all_images


def _load_caltech256(data_dir):
    """Load Caltech-256 via torchvision. Returns list of tensors."""
    print("\nLoading Caltech-256...")
    ds = torchvision.datasets.Caltech256(root=data_dir, download=True)
    all_images = []
    skipped = 0
    for i in range(len(ds)):
        img, _ = ds[i]
        tensor = _pil_to_tensor(img)
        if min(tensor.shape[1], tensor.shape[2]) >= MIN_IMAGE_DIM:
            all_images.append(_to_gray_u8(tensor))
        else:
            skipped += 1
    if skipped:
        print(f"  Skipped {skipped} images smaller than {MIN_IMAGE_DIM}px")
    print(f"Total Caltech-256 images: {len(all_images):,}")
    return all_images


def _load_imagenet(data_dir, num_samples):
    """Load ImageNet via HuggingFace streaming. Returns list of tensors.

    Only fetches as many images as needed (no 150GB download).
    Requires: pip install datasets, HuggingFace token with ImageNet access.
    """
    from datasets import load_dataset
    print(f"\nStreaming ImageNet from HuggingFace (fetching up to {num_samples:,})...")
    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
    all_images = []
    all_labels = []
    skipped = 0
    for item in ds:
        img = item["image"]
        tensor = _pil_to_tensor(img)
        if min(tensor.shape[1], tensor.shape[2]) >= MIN_IMAGE_DIM:
            all_images.append(_to_gray_u8(tensor))
            lbl = _record_label(item)
            all_labels.append(NO_LABEL if lbl is None else lbl)
        else:
            skipped += 1
        if len(all_images) >= num_samples:
            break
    if skipped:
        print(f"  Skipped {skipped} images smaller than {MIN_IMAGE_DIM}px")
    print(f"Total ImageNet images loaded: {len(all_images):,}")
    return all_images, all_labels, _label_names(ds)


def _record_image(item):
    """Pull the PIL image out of a dataset record, whatever its column
    name — parquet mirrors use 'image', webdataset mirrors 'jpg'/'webp'."""
    from PIL.Image import Image as PILImage
    for value in item.values():
        if isinstance(value, PILImage):
            return value
    return None


def _record_label(item):
    """Pull an integer class label (0..999) from a dataset record, or None."""
    for k in ("label", "cls", "fine_label", "class", "labels"):
        v = item.get(k) if hasattr(item, "get") else None
        if isinstance(v, int):
            return v
    return None


def _label_names(ds):
    """Human-readable class names from a dataset's features, or None.

    These become the text captions the model is conditioned on; ImageNet
    names list synonyms ("tench, Tinca tinca"), so keep only the first.
    """
    try:
        names = ds.features["label"].names
    except Exception:
        return None
    if not names:
        return None
    return [str(n).split(",")[0].strip() for n in names]


def _imagenet_class_names():
    """Packaged ImageNet-1k class names (line number = class id).

    Fallback for mirrors whose records carry integer labels but whose
    features have no name mapping — e.g. the webdataset ImageNet-Sketch
    mirrors, where losing the names would silently drop the captions.
    """
    path = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
    try:
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
    except OSError:
        return None
    return names if len(names) == 1000 else None


def _load_imagenet_sketch(num_samples):
    """Load ImageNet-Sketch via HuggingFace streaming. Returns list of tensors.

    ~50k black-on-white sketch renditions of the 1000 ImageNet classes.
    Line drawings convert to far cleaner ASCII than photos, and their
    distribution matches the stage-1 geometry data much better.

    The original repos are script-based, which datasets>=4 refuses to
    load — try the Hub's auto-converted parquet branch first, then a
    WebDataset mirror.
    """
    from datasets import load_dataset

    print(f"\nStreaming ImageNet-Sketch from HuggingFace "
          f"(fetching up to {num_samples:,})...")

    attempts = [
        ("songweig/imagenet_sketch @refs/convert/parquet",
         lambda: load_dataset("songweig/imagenet_sketch",
                              revision="refs/convert/parquet",
                              split="train", streaming=True)),
        ("imagenet_sketch @refs/convert/parquet",
         lambda: load_dataset("imagenet_sketch",
                              revision="refs/convert/parquet",
                              split="train", streaming=True)),
        ("songweig/imagenet_sketch parquet files (explicit glob)",
         lambda: load_dataset(
             "parquet", split="train", streaming=True,
             data_files="hf://datasets/songweig/imagenet_sketch"
                        "@refs/convert/parquet/default/*/*.parquet")),
        ("clip-benchmark/wds_imagenet_sketch (webdataset, test split)",
         lambda: load_dataset("clip-benchmark/wds_imagenet_sketch",
                              split="test", streaming=True)),
        ("clip-benchmark/wds_imagenet_sketch (webdataset, train split)",
         lambda: load_dataset("clip-benchmark/wds_imagenet_sketch",
                              split="train", streaming=True)),
    ]

    ds = None
    last_err = None
    for label, make in attempts:
        try:
            candidate = make()
            # Probe one real record: many loading failures only surface
            # on iteration, and a dud here must not cost a full pass
            if _record_image(next(iter(candidate))) is None:
                raise ValueError("no image column in first record")
            ds = candidate
            print(f"  Using: {label}")
            break
        except Exception as e:
            print(f"  ({label}: {type(e).__name__} — trying next mirror)")
            last_err = e
    if ds is None:
        raise RuntimeError(
            f"Could not load ImageNet-Sketch from any mirror: {last_err}")

    all_images = []
    all_labels = []
    skipped = 0
    t0 = time.time()
    for item in ds:  # streaming datasets restart cleanly after the probe
        img = _record_image(item)
        if img is None:
            skipped += 1
            continue
        tensor = _pil_to_tensor(img)
        if min(tensor.shape[1], tensor.shape[2]) >= MIN_IMAGE_DIM:
            all_images.append(_to_gray_u8(tensor))
            lbl = _record_label(item)
            all_labels.append(NO_LABEL if lbl is None else lbl)
            if len(all_images) % 5000 == 0:
                rate = len(all_images) / (time.time() - t0)
                print(f"  {len(all_images):>7,} images streamed "
                      f"({rate:.0f} img/s)", flush=True)
        else:
            skipped += 1
        if len(all_images) >= num_samples:
            break
    if skipped:
        print(f"  Skipped {skipped} records (too small or no image)")
    n_labeled = sum(1 for l in all_labels if l != NO_LABEL)
    print(f"Total ImageNet-Sketch images loaded: {len(all_images):,} "
          f"({n_labeled:,} with class labels)")
    return all_images, all_labels, _label_names(ds)


def _apply_segmentation(img_tensor):
    """Remove the background with rembg, compositing the subject on white.

    Optional dependency: pip install rembg onnxruntime
    """
    try:
        from rembg import remove
    except ImportError as e:
        raise ImportError(
            "Background segmentation requires rembg: "
            "pip install rembg onnxruntime") from e

    if img_tensor.dtype == torch.uint8:  # compact grayscale loader format
        arr = np.stack([img_tensor.numpy()] * 3, axis=-1)
    else:
        arr = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    out = remove(Image.fromarray(arr))  # RGBA
    rgba = np.asarray(out.convert("RGBA")).astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + (1.0 - alpha)  # composite on white
    return _to_gray_u8(torch.from_numpy(rgb).permute(2, 0, 1))


SOURCES = {
    "imagenet_sketch": "ImageNet-Sketch (HF streaming, ~50k line drawings)",
    "imagenet": "ImageNet (HuggingFace streaming, ~469x387 variable)",
    "imagenette": "Imagenette 320px (10 classes, ~13k images)",
    "caltech101": "Caltech-101 (101 classes, ~9k images)",
    "caltech256": "Caltech-256 (257 classes, ~30k images)",
    "stl10": "STL-10 (10 classes, ~113k images, 96x96) [legacy]",
}


def load_images(source, data_dir, num_samples):
    """Load images from a source. Returns (images, labels, label_names).

    labels is a list of class ids aligned with images and label_names maps
    those ids to caption strings, for the ImageNet-derived sources; both
    are None for sources without usable labels (they train unconditionally).
    """
    if source == "imagenet_sketch":
        return _load_imagenet_sketch(num_samples)
    elif source == "imagenet":
        return _load_imagenet(data_dir, num_samples)
    elif source == "stl10":
        return _load_stl10(os.path.join(data_dir, "stl10")), None, None
    elif source == "imagenette":
        return _load_imagenette(data_dir), None, None
    elif source == "caltech101":
        return _load_caltech101(data_dir), None, None
    elif source == "caltech256":
        return _load_caltech256(data_dir), None, None
    else:
        raise ValueError(f"Unknown source: {source}. Choose from: {list(SOURCES)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ASCII shading data from image datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available sources:\n" + "\n".join(
            f"  {k:12s}  {v}" for k, v in SOURCES.items()
        ),
    )
    parser.add_argument(
        "--source", default="imagenet_sketch", choices=list(SOURCES),
        help="Image dataset source (default: imagenet_sketch)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=SHADING_NUM_SAMPLES,
        help=f"Number of shading samples to generate (default: {SHADING_NUM_SAMPLES:,})",
    )
    parser.add_argument(
        "--segment", action="store_true",
        help="Remove image backgrounds with rembg before conversion "
             "(requires: pip install rembg onnxruntime)",
    )
    args = parser.parse_args()
    generate_dataset(args.source, args.num_samples, segment=args.segment)


def generate_dataset(source, num_samples, segment=False, out_path=None):
    """Generate the shading dataset and save it. Returns the [N, H, W] tensor."""
    t_start = time.time()
    data_dir = os.path.dirname(__file__)
    if out_path is None:
        out_path = os.path.join(data_dir, "shading_data.pt")

    # Build circle masks and character shape vectors
    print("Building 6D shape vectors...")
    masks = _make_circle_masks()
    mask_sums = np.array([m.sum() for m in masks], dtype=np.float32)
    shape_vectors, char_indices = compute_shape_vectors(masks)

    # Precompute squared norms for fast distance computation
    sv_sq_sum = (shape_vectors ** 2).sum(axis=1, keepdims=True).T  # [1, NUM_CHARS]

    print(f"\nGrid: {GRID_H}x{GRID_W}, Cell: {CELL_H}x{CELL_W}")
    print(f"Source: {source} — {SOURCES[source]}")
    print(f"Method: 6D shape vector matching ({NUM_CHARS} chars)")
    print(f"Pre-processing: adaptive_gamma (kernel={AG_KERNEL}), "
          f"CLAHE (clip={CLAHE_CLIP}, tiles={CLAHE_TILE_H}x{CLAHE_TILE_W})")
    print(f"Post-processing: edge_blend={EDGE_BLEND}, edge_boost={EDGE_BOOST}, "
          f"contrast_power={CONTRAST_POWER}, "
          f"whitespace_floor={WHITESPACE_FLOOR}")
    if segment:
        print("Segmentation: rembg background removal enabled")

    # Load images from selected source
    all_images, all_labels, label_names = load_images(source, data_dir,
                                                      num_samples)

    # Labels without a name mapping (webdataset mirrors): both labeled
    # sources use ImageNet-1k ids, so the packaged name list applies
    if all_labels is not None and label_names is None:
        packaged = _imagenet_class_names()
        valid = [l for l in all_labels if l >= 0]
        if packaged and valid and max(valid) < len(packaged):
            label_names = packaged
            print("  No class-name mapping in the dataset features — using "
                  "packaged ImageNet class names for captions")
    total_available = len(all_images)

    if total_available == 0:
        print("ERROR: No images loaded. Check dataset source and auth.")
        sys.exit(1)

    if total_available < num_samples:
        print(f"\n  Note: {total_available:,} images available, will cycle with "
              f"augmentation to reach {num_samples:,} samples")

    # Generate all samples
    print(f"\nGenerating {num_samples:,} shading samples...")
    samples = []
    sample_labels = [] if all_labels is not None else None
    t_gen = time.time()
    log_interval = 5_000
    segmented = set()

    for i in range(num_samples):
        idx = i % total_available
        img = all_images[idx]
        # Segment lazily and cache, so cycled images pay the cost once
        if segment and idx not in segmented:
            all_images[idx] = img = _apply_segmentation(img)
            segmented.add(idx)
        # Augment cycled images with random horizontal flip
        if i >= total_available and torch.rand(1).item() > 0.5:
            img = img.flip(-1)

        samples.append(image_to_ascii_grid(
            img, shape_vectors, masks, char_indices, mask_sums, sv_sq_sum))
        if sample_labels is not None:
            sample_labels.append(all_labels[idx])

        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t_gen
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  {i + 1:>7,}/{num_samples:,}  "
                  f"({100 * (i + 1) / num_samples:5.1f}%)  "
                  f"{rate:.0f} img/s  ETA {eta:.0f}s")

    gen_elapsed = time.time() - t_gen
    print(f"\nGeneration done in {gen_elapsed:.1f}s "
          f"({num_samples / gen_elapsed:.0f} img/s)")

    print("Stacking tensors...")
    # uint8: the 98-char vocab fits in a byte — 8x smaller file
    data = torch.stack(samples).to(torch.uint8)
    print(f"Tensor shape: {data.shape}, dtype: {data.dtype}")

    # Character usage stats
    unique, counts = torch.unique(data, return_counts=True)
    top_k = counts.argsort(descending=True)[:15]
    from data.charset import idx_to_char
    print(f"\nCharacter usage ({len(unique)} unique chars):")
    for idx in top_k:
        ch = idx_to_char(unique[idx].item())
        pct = 100 * counts[idx].item() / data.numel()
        print(f"  '{ch}' (idx {unique[idx].item():>2}): {pct:.1f}%")

    # Save as {data, caption_ids, captions} when class labels AND their
    # names are available (enables text-conditioned training + CFG); a
    # plain {data} dict otherwise.
    if (sample_labels is not None and label_names
            and any(l != NO_LABEL for l in sample_labels)):
        caption_ids = torch.tensor(sample_labels, dtype=torch.long)
        payload = {"data": data, "caption_ids": caption_ids,
                   "captions": label_names}
        n_captioned = int((caption_ids != NO_LABEL).sum())
        print(f"\nSaving to {out_path} with captions "
              f"({n_captioned:,}/{len(caption_ids):,} captioned, "
              f"{len(label_names)} unique class names)...")
    else:
        payload = {"data": data}
        print(f"\nSaving to {out_path} (unconditional, no captions)...")
    torch.save(payload, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved ({size_mb:.1f} MB)")

    total_elapsed = time.time() - t_start
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Print sample grids
    print("\n=== Sample Shading Grids ===\n")
    for i in range(min(5, len(samples))):
        print(f"--- Sample {i} ---")
        print(grid_to_string(samples[i]))
        print()

    return data


if __name__ == "__main__":
    main()
