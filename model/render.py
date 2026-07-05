"""Render character grids to images via the glyph atlas.

The bridge between glyph space and pixel space — ASCII art is judged by
how it renders, and this makes the rendering available to code:

- render_grid(grid): hard render of token indices, used to show grids to
  CLIP/VLMs (candidate re-ranking, auto-captioning).
- render_probs(probs): soft render of a probability grid — a weighted sum
  of glyph bitmaps, differentiable w.r.t. the probabilities. This is the
  hook for future perceptual-loss training (CLIP/VLM rewards on renders).

Both are pure tensor ops over a cached [VOCAB_SIZE, CELL_H, CELL_W] atlas
built from the same rasterizer as the glyph-similarity soft labels, so
the loss-side and render-side views of each glyph agree.
"""

import torch

from config import VOCAB_SIZE

_ATLAS = None


def glyph_atlas():
    """[VOCAB_SIZE, CELL_H, CELL_W] float bitmaps in [0,1] (ink=1).

    PAD/MASK render blank. Cached after first build. Falls back to an
    all-blank atlas if no font is available (render becomes useless but
    nothing crashes — callers that care should check atlas().sum() > 0).
    """
    global _ATLAS
    if _ATLAS is None:
        from data.charset import char_to_idx
        from data.glyph_sim import _CELL_H, _CELL_W, _PRINTABLE, \
            _glyph_bitmaps
        atlas = torch.zeros(VOCAB_SIZE, _CELL_H, _CELL_W)
        bmps = _glyph_bitmaps()
        if bmps is not None:
            tiles = torch.from_numpy(bmps).view(len(_PRINTABLE), _CELL_H,
                                                _CELL_W)
            for i, ch in enumerate(_PRINTABLE):
                atlas[char_to_idx(ch)] = tiles[i]
        _ATLAS = atlas
    return _ATLAS


def render_grid(grid):
    """[H, W] long tensor -> [H*CELL_H, W*CELL_W] float image, ink=1."""
    atlas = glyph_atlas().to(grid.device)
    tiles = atlas[grid]                      # [H, W, ch, cw]
    H, W, ch, cw = tiles.shape
    return tiles.permute(0, 2, 1, 3).reshape(H * ch, W * cw)


def render_probs(probs):
    """[H, W, VOCAB_SIZE] probabilities -> soft render (differentiable).

    A one-hot input reproduces render_grid exactly; a soft input renders
    the expectation over glyphs, which is what lets image-space losses
    backpropagate into glyph logits.
    """
    atlas = glyph_atlas().to(device=probs.device, dtype=probs.dtype)
    tiles = torch.einsum("hwv,vij->hwij", probs, atlas)
    H, W, ch, cw = tiles.shape
    return tiles.permute(0, 2, 1, 3).reshape(H * ch, W * cw)


def render_to_pil(grid, invert=True):
    """Render a grid to a PIL RGB image (black ink on white by default —
    the polarity CLIP-family models saw line drawings in)."""
    from PIL import Image
    img = render_grid(grid).clamp(0, 1)
    if invert:
        img = 1.0 - img
    arr = (img * 255).byte().cpu().numpy()
    return Image.fromarray(arr).convert("RGB")
