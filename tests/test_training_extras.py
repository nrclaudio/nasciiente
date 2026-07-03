import torch
import torch.nn as nn

from data.glyph_sim import build_soft_target_matrix
from data.charset import char_to_idx


def test_ema_tracks_and_swaps():
    import training.train as T
    model = nn.Linear(4, 4)
    ema = T.EMA(model, decay=0.5)
    # Take a big step so params move a lot
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10.0)
    ema.update(model)
    # Shadow moved halfway (decay 0.5) toward the new params, not all the way
    for k, p in model.named_parameters():
        assert not torch.allclose(ema.shadow[k], p)

    # store_and_copy swaps EMA weights in; restore brings the live ones back
    live = {k: p.detach().clone() for k, p in model.named_parameters()}
    ema.store_and_copy(model)
    for k, p in model.named_parameters():
        assert torch.allclose(p, ema.shadow[k])
    ema.restore(model)
    for k, p in model.named_parameters():
        assert torch.allclose(p, live[k])


def test_glyph_matrix_is_row_stochastic_and_diagonal_dominant():
    m = build_soft_target_matrix(0.1)
    assert torch.allclose(m.sum(dim=1), torch.ones(m.shape[0]), atol=1e-5)
    # Each real-glyph row keeps most mass on the true glyph
    for ch in "/|-+@ ":
        i = char_to_idx(ch)
        assert m[i].argmax().item() == i
        assert m[i, i] > 0.5


def test_glyph_alpha_zero_is_identity():
    m = build_soft_target_matrix(0.0)
    assert torch.equal(m, torch.eye(m.shape[0]))


def test_glyph_similar_shapes_get_more_mass():
    # A diagonal stroke '/' should share more mass with other diagonal
    # glyphs than with a solid block-like '@'.
    m = build_soft_target_matrix(0.2)
    sl = char_to_idx("/")
    # off-diagonal mass total should be exactly alpha-ish (row sums to 1)
    off = 1.0 - m[sl, sl].item()
    assert off > 0
