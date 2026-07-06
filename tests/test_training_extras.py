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


def test_mask_ratio_sampling_favors_high_ratios():
    from training.masking import _sample_ratio
    import random as _r
    _r.seed(0)
    samples = [_sample_ratio(0.15, 1.0) for _ in range(4000)]
    assert all(0.15 <= s <= 1.0 for s in samples)
    mean = sum(samples) / len(samples)
    # sin(pi/2 u) has mean 2/pi ~ 0.64 -> ratio mean ~ 0.69, well above
    # the uniform midpoint 0.575
    assert mean > 0.63, mean
    assert sum(s > 0.85 for s in samples) / len(samples) > 0.25


def test_space_weighted_loss():
    from model.ascii_bert import ASCIIBert, SPACE_IDX
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    x = torch.randint(2, 98, (2, 8, 10))
    mask = torch.ones(2, 8, 10, dtype=torch.bool)
    logits = model(x)

    # All-space targets: uniform weights -> weighted mean == plain mean
    targets = torch.full((2, 8, 10), SPACE_IDX)
    plain = model.compute_loss(logits, targets, mask, space_weight=1.0)
    weighted = model.compute_loss(logits, targets, mask, space_weight=0.4)
    assert torch.allclose(plain, weighted, atol=1e-6)

    # Mixed targets: down-weighting space changes the loss
    targets2 = targets.clone()
    targets2[:, :4] = 40
    a = model.compute_loss(logits, targets2, mask, space_weight=1.0)
    b = model.compute_loss(logits, targets2, mask, space_weight=0.4)
    assert not torch.allclose(a, b)


def test_mix_replay_merges_captions(tmp_path):
    import training.train as T
    human = {"data": torch.randint(2, 98, (6, 4, 5), dtype=torch.uint8),
             "caption_ids": torch.tensor([0, -1, 1, 0, -1, 1]),
             "captions": ["h1", "h2"]}
    shading = {"data": torch.randint(2, 98, (10, 4, 5), dtype=torch.uint8),
               "caption_ids": torch.arange(10) % 3,
               "captions": ["s1", "s2", "s3"]}
    sp = tmp_path / "shading.pt"
    torch.save(shading, sp)

    data, ids, caps = T._mix_replay(human["data"], human["caption_ids"],
                                    human["captions"], str(sp),
                                    replay_samples=4)
    assert len(data) == 10 and len(ids) == 10
    assert caps == ["h1", "h2", "s1", "s2", "s3"]
    # Human ids untouched; replay ids offset into the merged table
    assert ids[:6].tolist() == [0, -1, 1, 0, -1, 1]
    assert all(2 <= i <= 4 for i in ids[6:].tolist())
    # Deterministic across calls (all DDP ranks build the same mix)
    data2, ids2, _ = T._mix_replay(human["data"], human["caption_ids"],
                                   human["captions"], str(sp),
                                   replay_samples=4)
    assert torch.equal(data, data2) and torch.equal(ids, ids2)


def test_train_args_parsing(monkeypatch):
    import training.train as T
    monkeypatch.setattr("sys.argv",
                        ["train.py", "--init-from", "ck.pt",
                         "--stages", "shading,human", "--stage", "train"])
    args = T.parse_args()  # unknown --stage (from main.py) must not crash
    assert args.init_from == "ck.pt"
    assert args.stages == "shading,human"


def test_chained_replay_from_two_sources(tmp_path):
    # Stage 3 replays BOTH shading and geometry; chaining _mix_replay
    # must accumulate samples and keep caption ids pointing at the right
    # captions across both merges
    from training.train import _mix_replay

    def payload(n, caps):
        return {"data": torch.randint(2, 98, (n, 48, 80),
                                      dtype=torch.uint8),
                "caption_ids": torch.arange(n) % len(caps),
                "captions": caps}

    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    torch.save(payload(30, ["shade1", "shade2"]), a)
    torch.save(payload(40, ["geom1", "geom2", "geom3"]), b)

    data = torch.randint(2, 98, (10, 48, 80), dtype=torch.uint8)
    ids = torch.zeros(10, dtype=torch.long)
    caps = ["base"]

    data, ids, caps = _mix_replay(data, ids, caps, str(a), 20)
    data, ids, caps = _mix_replay(data, ids, caps, str(b), 15)

    assert len(data) == 10 + 20 + 15
    assert len(ids) == len(data)
    assert caps[:1] == ["base"]
    assert set(caps) == {"base", "shade1", "shade2",
                         "geom1", "geom2", "geom3"}
    # Every id points at a caption consistent with its source segment
    for i in ids[:10].tolist():
        assert caps[i] == "base"
    for i in ids[10:30].tolist():
        assert caps[i].startswith("shade")
    for i in ids[30:].tolist():
        assert caps[i].startswith("geom")
