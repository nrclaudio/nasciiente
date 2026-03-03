import random

import torch

from data.charset import MASK_TOKEN
from config import MASK_RATIO_MIN, MASK_RATIO_MAX


def random_mask(grid, mask_ratio_min=MASK_RATIO_MIN, mask_ratio_max=MASK_RATIO_MAX):
    """
    Randomly mask a percentage of positions in a grid.

    Args:
        grid: [H, W] long tensor of character indices
        mask_ratio_min: minimum fraction to mask
        mask_ratio_max: maximum fraction to mask

    Returns:
        masked_grid: [H, W] with some positions replaced by MASK_TOKEN
        target_grid: [H, W] original grid (for loss computation)
        mask: [H, W] bool tensor (True where masked)
    """
    H, W = grid.shape
    target_grid = grid.clone()

    ratio = random.uniform(mask_ratio_min, mask_ratio_max)
    num_to_mask = int(H * W * ratio)

    # Random permutation to select positions
    perm = torch.randperm(H * W)[:num_to_mask]
    mask = torch.zeros(H * W, dtype=torch.bool)
    mask[perm] = True
    mask = mask.view(H, W)

    masked_grid = grid.clone()
    masked_grid[mask] = MASK_TOKEN

    return masked_grid, target_grid, mask
