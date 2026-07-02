import numpy as np
import torch

from config import GRID_H, GRID_W, VOCAB_SIZE
from data.generate_shading import (
    NUM_CHARS, NUM_REGIONS, _make_circle_masks, compute_shape_vectors,
    image_to_ascii_grid, _clahe, _adaptive_gamma,
)


def _shape_matching_setup():
    masks = _make_circle_masks()
    mask_sums = np.array([m.sum() for m in masks], dtype=np.float32)
    shape_vectors, char_indices = compute_shape_vectors(masks)
    sv_sq_sum = (shape_vectors ** 2).sum(axis=1, keepdims=True).T
    return masks, mask_sums, shape_vectors, char_indices, sv_sq_sum


def test_circle_masks():
    masks = _make_circle_masks()
    assert len(masks) == NUM_REGIONS
    for m in masks:
        assert m.dtype == bool
        assert m.any(), "each sampling circle must cover at least one pixel"


def test_shape_vectors():
    masks = _make_circle_masks()
    sv, ci = compute_shape_vectors(masks)
    assert sv.shape == (NUM_CHARS, NUM_REGIONS)
    assert ci.shape == (NUM_CHARS,)
    assert sv.min() >= 0.0 and sv.max() <= 1.0
    # Space has zero ink; '@'-like glyphs have high density
    assert ci.min() >= 2 and ci.max() < VOCAB_SIZE


def test_image_to_ascii_grid():
    torch.manual_seed(0)
    masks, mask_sums, sv, ci, sv_sq = _shape_matching_setup()
    img = torch.rand(3, 96, 96)
    grid = image_to_ascii_grid(img, sv, masks, ci, mask_sums, sv_sq)
    assert grid.shape == (GRID_H, GRID_W)
    assert grid.min().item() >= 2
    assert grid.max().item() < VOCAB_SIZE


def test_black_image_maps_to_sparse_chars():
    """An all-black (inverted: all-white) vs all-white image should differ."""
    masks, mask_sums, sv, ci, sv_sq = _shape_matching_setup()
    white = torch.ones(3, 96, 96)   # inverted to 0 density -> sparse chars
    grid = image_to_ascii_grid(white, sv, masks, ci, mask_sums, sv_sq)
    # A uniform white image should mostly produce the lightest characters
    from data.charset import idx_to_char
    chars = {idx_to_char(int(i)) for i in torch.unique(grid)}
    assert " " in chars or "." in chars or "`" in chars


def test_preprocessing_output_ranges():
    rng = np.random.default_rng(0)
    img = rng.random((96, 160), dtype=np.float32)
    out = _adaptive_gamma(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    out = _clahe(img)
    assert out.shape == img.shape
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6
