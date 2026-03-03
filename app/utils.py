import torch

from data.charset import MASK_TOKEN, char_to_idx, string_to_grid


def text_to_partial_grid(text, grid_h, grid_w):
    """
    Convert user text input to a partial grid for inpainting.

    Spaces remain as spaces (not masked). Empty lines / short lines
    are padded with MASK tokens.

    Args:
        text: multi-line string from user
        grid_h: target grid height
        grid_w: target grid width

    Returns:
        [grid_h, grid_w] long tensor with MASK_TOKEN for empty positions
    """
    grid = torch.full((grid_h, grid_w), MASK_TOKEN, dtype=torch.long)
    lines = text.split("\n")

    for r, line in enumerate(lines):
        if r >= grid_h:
            break
        for c, ch in enumerate(line):
            if c >= grid_w:
                break
            try:
                grid[r, c] = char_to_idx(ch)
            except KeyError:
                grid[r, c] = MASK_TOKEN

    return grid


def count_fixed_positions(grid):
    """Count non-MASK positions in a grid."""
    return (grid != MASK_TOKEN).sum().item()
