import random

import torch

from data.charset import MASK_TOKEN
from config import MASK_RATIO_MIN, MASK_RATIO_MAX


def _apply(grid, mask):
    """Build (masked_grid, target_grid, mask, ratio) from a boolean mask."""
    target_grid = grid.clone()
    masked_grid = grid.clone()
    masked_grid[mask] = MASK_TOKEN
    ratio = mask.float().mean().item()
    return masked_grid, target_grid, mask, ratio


def random_mask(grid, mask_ratio_min=MASK_RATIO_MIN, mask_ratio_max=MASK_RATIO_MAX):
    """Mask a uniform-random subset of cells (the original scheme).

    Returns (masked_grid, target_grid, mask, ratio).
    """
    H, W = grid.shape
    ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    num_to_mask = int(H * W * ratio)
    perm = torch.randperm(H * W)[:num_to_mask]
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[perm] = True
    return _apply(grid, mask.view(H, W))


def block_mask(grid, mask_ratio_min=MASK_RATIO_MIN, mask_ratio_max=MASK_RATIO_MAX):
    """Mask a single contiguous rectangle — mimics the inpainting use case.

    Training on this makes the app's "fill this region" task in-distribution
    rather than something the random-cell scheme only approximates.
    """
    H, W = grid.shape
    ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    # Rectangle area ~= ratio * H * W, with a randomized aspect ratio
    area = ratio * H * W
    aspect = random.uniform(0.5, 2.0)
    bh = min(H, max(1, int(round((area * aspect) ** 0.5))))
    bw = min(W, max(1, int(round(area / bh))))
    r0 = random.randint(0, H - bh)
    c0 = random.randint(0, W - bw)
    mask = torch.zeros(H, W, dtype=torch.bool)
    mask[r0:r0 + bh, c0:c0 + bw] = True
    return _apply(grid, mask)


def anchor_mask(grid, stride=None):
    """Keep a strided lattice of anchor cells, mask everything else.

    This is exactly the pattern upscale_grid() creates (source chars at
    positions (stride*r, stride*c), gaps masked). Training on it makes
    super-resolution in-distribution instead of hopeful extrapolation.
    """
    H, W = grid.shape
    if stride is None:
        stride = random.choice([2, 2, 3])  # favor 2x
    mask = torch.ones(H, W, dtype=torch.bool)
    mask[::stride, ::stride] = False  # anchors are kept (not masked)
    return _apply(grid, mask)


# Sampling weights for the mask mode used on each training example.
MASK_MODES = ("random", "block", "anchor")
MASK_MODE_WEIGHTS = (0.7, 0.15, 0.15)


def mixed_mask(grid, mask_ratio_min=MASK_RATIO_MIN, mask_ratio_max=MASK_RATIO_MAX):
    """Pick a masking scheme per example so the model learns all inference
    modes (free generation, inpainting, upscaling) from one training run.

    Returns (masked_grid, target_grid, mask, ratio).
    """
    mode = random.choices(MASK_MODES, weights=MASK_MODE_WEIGHTS, k=1)[0]
    if mode == "block":
        return block_mask(grid, mask_ratio_min, mask_ratio_max)
    if mode == "anchor":
        return anchor_mask(grid)
    return random_mask(grid, mask_ratio_min, mask_ratio_max)
