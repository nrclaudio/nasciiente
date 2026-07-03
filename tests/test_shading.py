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


def test_uniform_backgrounds_render_as_whitespace():
    """Flat images are all background — whitespace floor makes them blank."""
    from data.charset import char_to_idx
    masks, mask_sums, sv, ci, sv_sq = _shape_matching_setup()
    space = char_to_idx(" ")
    white = torch.ones(3, 96, 96)
    grid = image_to_ascii_grid(white, sv, masks, ci, mask_sums, sv_sq)
    assert (grid == space).all(), "white image should be pure whitespace"
    # Dark image: auto-polarity flips so the background is still sparse
    black = torch.zeros(3, 96, 96)
    grid = image_to_ascii_grid(black, sv, masks, ci, mask_sums, sv_sq)
    assert (grid == space).all(), "black image should be pure whitespace"


def test_subject_renders_without_background_halo():
    """A dark disk on white: dense subject chars, clean whitespace around."""
    from data.charset import char_to_idx
    masks, mask_sums, sv, ci, sv_sq = _shape_matching_setup()
    space = char_to_idx(" ")
    img = torch.ones(3, 480, 480)
    yy, xx = torch.meshgrid(torch.arange(480), torch.arange(480),
                            indexing="ij")
    disk = ((yy - 240) ** 2 + (xx - 240) ** 2) < 120 ** 2
    img[:, disk] = 0.0

    grid = image_to_ascii_grid(img, sv, masks, ci, mask_sums, sv_sq)
    assert (grid != space).sum().item() > 50, "disk should render as ink"
    # Corners are far from the disk — must be pure whitespace, no halo
    assert (grid[:4, :6] == space).all()
    assert (grid[-4:, -6:] == space).all()


def test_letterbox_preserves_aspect_ratio():
    """A wide image gets whitespace bands top and bottom, not stretched."""
    from data.charset import char_to_idx
    masks, mask_sums, sv, ci, sv_sq = _shape_matching_setup()
    space = char_to_idx(" ")
    img = torch.zeros(3, 100, 800)  # very wide, all-dark -> polarity flips

    # Force ink everywhere in-image: dark subject strip on white, wide image
    img = torch.ones(3, 100, 800)
    img[:, 20:80, :] = 0.0
    grid = image_to_ascii_grid(img, sv, masks, ci, mask_sums, sv_sq)
    # 100x800 image inside 576x640 canvas -> content is ~80px tall (~7 rows),
    # centered; the top and bottom thirds of the grid must be blank
    assert (grid[:12] == space).all()
    assert (grid[-12:] == space).all()
    assert (grid != space).any()


def test_preprocessing_output_ranges():
    rng = np.random.default_rng(0)
    img = rng.random((96, 160), dtype=np.float32)
    out = _adaptive_gamma(img)
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    out = _clahe(img)
    assert out.shape == img.shape
    assert out.min() >= -1e-6 and out.max() <= 1.0 + 1e-6


def test_record_image_finds_pil_under_any_column():
    from PIL import Image
    from data.generate_shading import _record_image
    img = Image.new("L", (8, 8))
    assert _record_image({"jpg": img, "cls": 3}) is img
    assert _record_image({"image": img}) is img
    assert _record_image({"label": 1, "text": "x"}) is None
