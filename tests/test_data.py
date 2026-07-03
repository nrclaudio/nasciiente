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


def test_distributed_defaults_without_torchrun():
    """train.py must run single-process when torchrun env vars are absent."""
    import training.train as T
    assert "RANK" not in __import__("os").environ
    assert T.setup_distributed() == (0, 1, 0)
    # unwrap is a no-op on a bare module
    import torch.nn as nn
    m = nn.Linear(2, 2)
    assert T.unwrap(m) is m


def test_lr_scaling_with_world_size(monkeypatch):
    import training.train as T
    assert T.scale_lr(3e-4) == 3e-4  # single process: unchanged
    monkeypatch.setattr(T, "WORLD_SIZE", 4)
    assert T.scale_lr(3e-4) == 3e-4 * 4
    monkeypatch.setattr(T, "SCALE_LR_WITH_GPUS", False)
    assert T.scale_lr(3e-4) == 3e-4
