import torch

from app.utils import text_to_partial_grid, count_fixed_positions
from data.charset import MASK_TOKEN, char_to_idx


def test_text_to_partial_grid_basic():
    grid = text_to_partial_grid("Hi\nthere", 10, 20)
    assert grid.shape == (10, 20)
    assert count_fixed_positions(grid) == 7
    assert grid[0, 0].item() == char_to_idx("H")
    assert grid[1, 4].item() == char_to_idx("e")
    # Everything else stays masked
    assert grid[2, 0].item() == MASK_TOKEN
    assert grid[0, 2].item() == MASK_TOKEN


def test_text_to_partial_grid_truncates():
    text = "\n".join(["x" * 50] * 20)
    grid = text_to_partial_grid(text, 5, 10)
    assert grid.shape == (5, 10)
    assert count_fixed_positions(grid) == 50


def test_unknown_chars_become_mask():
    grid = text_to_partial_grid("aéb", 2, 10)  # é is not printable ASCII
    assert grid[0, 0].item() == char_to_idx("a")
    assert grid[0, 1].item() == MASK_TOKEN
    assert grid[0, 2].item() == char_to_idx("b")
