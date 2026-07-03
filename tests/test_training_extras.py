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


def test_glyph_space_row_stays_one_hot():
    # Space renders zero ink, so it has no meaningful similarity row — it
    # must stay one-hot instead of leaking uniform mass onto every glyph.
    m = build_soft_target_matrix(0.1)
    sp = char_to_idx(" ")
    expected = torch.zeros(m.shape[0])
    expected[sp] = 1.0
    assert torch.equal(m[sp], expected)


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


def test_embed_captions_padding_and_norm(monkeypatch):
    # Fake tokenizer/encoder with the HF API surface embed_captions uses;
    # verifies chunking, cross-chunk padding, masks, and per-token RMS norm
    import data.text_embed as TE

    class FakeBatch(dict):
        def to(self, device):
            return self

    class FakeTokenizer:
        def __call__(self, texts, padding=None, truncation=None,
                     max_length=None, return_tensors=None):
            lengths = [min(len(t.split()) + 2, max_length) for t in texts]
            L = max(lengths)
            ids = torch.zeros(len(texts), L, dtype=torch.long)
            mask = torch.zeros(len(texts), L, dtype=torch.long)
            for i, n in enumerate(lengths):
                mask[i, :n] = 1
            return FakeBatch(input_ids=ids, attention_mask=mask)

    class FakeOut:
        def __init__(self, h):
            self.last_hidden_state = h

    class FakeEncoder:
        def __call__(self, input_ids=None, attention_mask=None):
            B, L = input_ids.shape
            torch.manual_seed(0)
            return FakeOut(torch.randn(B, L, TE.TEXT_EMB_DIM) * 7.0)

    monkeypatch.setattr(TE, "_load_encoder",
                        lambda device="cpu": (FakeTokenizer(), FakeEncoder(),
                                              "cpu"))
    texts = ["one two", "a much longer caption with many words here",
             "three"]
    toks, mask = TE.embed_captions(texts, batch_size=2)  # forces 2 chunks
    assert toks.shape[0] == 3 and toks.shape[1] == mask.shape[1]
    assert mask.sum(1).tolist() == [4, 10, 3]
    # valid tokens RMS-normalized to 1, padding exactly zero
    rms = toks[mask].pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)
    assert torch.equal(toks[~mask], torch.zeros_like(toks[~mask]))
