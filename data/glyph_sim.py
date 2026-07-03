"""Glyph-aware soft targets.

Builds a [VOCAB_SIZE, VOCAB_SIZE] matrix whose row t is the training target
distribution for true token t: mostly one-hot, but a little probability mass
spread onto *visually similar* glyphs. So predicting '|' when the answer is
'/' costs less than predicting '@' — the model learns the visual structure
of the character set instead of treating 95 glyphs as arbitrary symbols.

The similarity comes from rendering each glyph to a small bitmap (the same
idea the shading pipeline already uses) and comparing bitmaps. Degrades to
an identity matrix (plain cross-entropy) if no font is available.
"""
import os

import numpy as np
import torch

from config import VOCAB_SIZE
from data.charset import char_to_idx

_PRINTABLE = [chr(c) for c in range(32, 127)]
_CELL_H, _CELL_W = 12, 8


def _find_mono_font(size=32):
    from PIL import ImageFont
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _glyph_bitmaps():
    """[num_printable, CELL_H*CELL_W] float bitmaps, or None if no font."""
    try:
        from PIL import Image, ImageDraw
        font = _find_mono_font(32)
        bbox = font.getbbox("M")
        fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if fw <= 0 or fh <= 0:
            return None
        bmps = []
        for ch in _PRINTABLE:
            img = Image.new("L", (fw, fh), 0)
            ImageDraw.Draw(img).text((-bbox[0], -bbox[1]), ch, fill=255,
                                     font=font)
            img = img.resize((_CELL_W, _CELL_H), Image.LANCZOS)
            bmps.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
        return np.stack(bmps)
    except Exception:
        return None


def build_soft_target_matrix(alpha, temperature=0.15, vocab_size=VOCAB_SIZE):
    """Return a [vocab_size, vocab_size] row-stochastic target matrix.

    Row t = (1 - alpha) * onehot(t) + alpha * softmax(similarity to t).
    alpha == 0 (or no font) yields the identity matrix (plain CE).
    """
    eye = torch.eye(vocab_size)
    if alpha <= 0:
        return eye
    bmps = _glyph_bitmaps()
    if bmps is None:
        return eye

    x = torch.from_numpy(bmps)                      # [P, D]
    norms = x.norm(dim=1, keepdim=True)
    x = x / (norms + 1e-8)
    sim = x @ x.t()                                 # cosine, [P, P]

    idx = torch.tensor([char_to_idx(c) for c in _PRINTABLE])  # vocab indices
    matrix = eye.clone()
    weights = torch.softmax(sim / temperature, dim=1)         # [P, P]
    # Zero-ink glyphs (space) have an all-zero similarity row, which would
    # softmax to a uniform distribution over every glyph — keep them one-hot
    blank = norms.squeeze(1) == 0
    for i, vi in enumerate(idx):
        if blank[i]:
            continue
        row = torch.zeros(vocab_size)
        row[idx] = weights[i]
        matrix[vi] = (1 - alpha) * eye[vi] + alpha * row
    return matrix
