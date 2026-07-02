import torch

from config import GRID_H, GRID_W, VOCAB_SIZE
from data.charset import MASK_TOKEN
from data.generate_geometry import generate_sample, PRIMITIVES, _blank_grid
from training.dataset import ASCIIDataset
from training.masking import random_mask


def test_geometry_sample_shape_and_range():
    torch.manual_seed(0)
    for _ in range(20):
        grid = generate_sample()
        assert grid.shape == (GRID_H, GRID_W)
        assert grid.dtype == torch.long
        # Only printable chars (indices 2..97) — never PAD or MASK
        assert grid.min().item() >= 2
        assert grid.max().item() < VOCAB_SIZE


def test_each_primitive_draws_in_bounds():
    for fn in PRIMITIVES:
        for _ in range(10):
            grid = _blank_grid()
            fn(grid)  # would raise IndexError if out of bounds
            assert grid.shape == (GRID_H, GRID_W)
            assert grid.max().item() < VOCAB_SIZE


def test_random_mask_invariants():
    grid = generate_sample()
    masked, target, mask = random_mask(grid)
    assert (target == grid).all()
    assert (masked[mask] == MASK_TOKEN).all()
    assert (masked[~mask] == grid[~mask]).all()
    ratio = mask.float().mean().item()
    assert 0.10 < ratio < 0.90


def test_dataset_getitem():
    data = torch.stack([generate_sample() for _ in range(4)])
    ds = ASCIIDataset(data)
    assert len(ds) == 4
    masked, target, mask = ds[0]
    assert masked.shape == target.shape == mask.shape == (GRID_H, GRID_W)
    assert mask.dtype == torch.bool
