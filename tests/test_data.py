import torch

from config import GRID_H, GRID_W, VOCAB_SIZE
from data.charset import MASK_TOKEN
from data.generate_geometry import generate_sample, PRIMITIVES, _blank_grid
from training.dataset import ASCIIDataset
from training.masking import (
    random_mask, block_mask, anchor_mask, mixed_mask,
)


def test_geometry_sample_shape_and_range():
    torch.manual_seed(0)
    for _ in range(20):
        grid, caption = generate_sample()
        assert grid.shape == (GRID_H, GRID_W)
        assert grid.dtype == torch.long
        # Only printable chars (indices 2..97) — never PAD or MASK
        assert grid.min().item() >= 2
        assert grid.max().item() < VOCAB_SIZE
        assert isinstance(caption, str) and caption


def test_each_primitive_draws_in_bounds():
    for fn in PRIMITIVES:
        for _ in range(10):
            grid = _blank_grid()
            fn(grid)  # would raise IndexError if out of bounds
            assert grid.shape == (GRID_H, GRID_W)
            assert grid.max().item() < VOCAB_SIZE


def test_random_mask_invariants():
    grid, _ = generate_sample()
    masked, target, mask, ratio = random_mask(grid)
    assert (target == grid).all()
    assert (masked[mask] == MASK_TOKEN).all()
    assert (masked[~mask] == grid[~mask]).all()
    assert abs(mask.float().mean().item() - ratio) < 1e-6
    assert 0.10 < ratio <= 1.0  # training covers up to fully masked


def test_block_mask_is_contiguous_rectangle():
    grid, _ = generate_sample()
    masked, target, mask, ratio = block_mask(grid)
    assert (masked[mask] == MASK_TOKEN).all()
    rows = mask.any(dim=1).nonzero().flatten()
    cols = mask.any(dim=0).nonzero().flatten()
    # Every cell in the bounding box is masked -> a solid rectangle
    box = mask[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    assert box.all()


def test_anchor_mask_keeps_strided_lattice():
    grid, _ = generate_sample()
    for stride in (2, 3):
        masked, target, mask, ratio = anchor_mask(grid, stride=stride)
        # Anchor cells are NOT masked and keep their original value
        assert not mask[::stride, ::stride].any()
        assert (masked[::stride, ::stride] == grid[::stride, ::stride]).all()


def test_mixed_mask_returns_valid_tuple():
    grid, _ = generate_sample()
    for _ in range(20):
        masked, target, mask, ratio = mixed_mask(grid)
        assert (masked[mask] == MASK_TOKEN).all()
        assert (masked[~mask] == grid[~mask]).all()
        assert 0.0 <= ratio <= 1.0


def test_dataset_getitem():
    data = torch.stack([generate_sample()[0] for _ in range(4)])
    caption_ids = torch.tensor([1, 0, -1, 1])
    caption_tokens = torch.randn(2, 6, 8)
    caption_masks = torch.tensor([[True] * 4 + [False] * 2,
                                  [True] * 6])
    ds = ASCIIDataset(data, caption_ids, caption_tokens, caption_masks)
    assert len(ds) == 4
    masked, target, mask, ratio, toks, tmask, has_cond = ds[0]
    assert masked.shape == target.shape == mask.shape == (GRID_H, GRID_W)
    assert mask.dtype == torch.bool
    assert ratio.dtype == torch.float
    assert bool(has_cond)
    assert torch.allclose(toks, caption_tokens[1])
    assert torch.equal(tmask, caption_masks[1])
    # Uncaptioned sample (caption_id -1): zero tokens, all-False mask
    _, _, _, _, toks2, tmask2, has_cond2 = ds[2]
    assert not bool(has_cond2)
    assert torch.equal(toks2, torch.zeros(6, 8))
    assert not tmask2.any()


def test_dataset_without_captions_is_unconditional():
    from config import TEXT_EMB_DIM
    data = torch.stack([generate_sample()[0] for _ in range(2)])
    ds = ASCIIDataset(data)
    _, _, _, _, toks, tmask, has_cond = ds[0]
    assert not bool(has_cond)
    assert torch.equal(toks, torch.zeros(1, TEXT_EMB_DIM))
    assert not tmask.any()


def test_load_data_and_captions_formats(tmp_path):
    import training.train as T
    grids = torch.randint(2, 98, (4, 8, 8), dtype=torch.uint8)

    bare = tmp_path / "bare.pt"
    torch.save(grids, bare)
    data, ids, caps = T._load_data_and_captions(str(bare))
    assert torch.equal(data, grids) and ids is None and caps is None

    legacy = tmp_path / "legacy.pt"  # interim class-label format
    torch.save({"data": grids, "labels": torch.tensor([1, 2, 3, 4])}, legacy)
    data, ids, caps = T._load_data_and_captions(str(legacy))
    assert torch.equal(data, grids) and ids is None and caps is None

    new = tmp_path / "new.pt"
    torch.save({"data": grids, "caption_ids": torch.tensor([0, 1, -1, 0]),
                "captions": ["a cat", "a dog"]}, new)
    data, ids, caps = T._load_data_and_captions(str(new))
    assert torch.equal(data, grids)
    assert ids.tolist() == [0, 1, -1, 0]
    assert caps == ["a cat", "a dog"]


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
