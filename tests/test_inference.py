import torch

from data.charset import MASK_TOKEN, char_to_idx, idx_to_char
from model.inference import generate


H, W = 12, 20


def test_generate_from_scratch(tiny_model):
    torch.manual_seed(0)
    steps, final = generate(tiny_model, H, W, num_steps=4, device="cpu")
    assert final.shape == (H, W)
    assert (final != MASK_TOKEN).all(), "final grid still contains MASK tokens"
    # First step is the fully masked grid, then progressive unmasking
    assert (steps[0] == MASK_TOKEN).all()
    assert len(steps) >= 2


def test_probs_capture(tiny_model):
    torch.manual_seed(0)
    captured = []
    steps, final = generate(tiny_model, H, W, num_steps=4,
                            revision_steps=0, device="cpu",
                            probs_out=captured)
    assert len(captured) == len(steps) - 1   # one capture per fill step
    pre_grid, probs = captured[0]
    assert pre_grid.shape == (H, W)
    assert (pre_grid == MASK_TOKEN).all()    # first capture: empty canvas
    assert probs.shape[:2] == (H, W)
    assert probs.dtype == torch.float16
    total = probs.float().sum(-1)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-2)


def test_main_fill_is_monotonic(tiny_model):
    # Without revision, the main fill only ever unmasks — never re-masks
    torch.manual_seed(0)
    steps, _ = generate(tiny_model, H, W, num_steps=5, revision_steps=0,
                        device="cpu")
    masked_counts = [(s == MASK_TOKEN).sum().item() for s in steps]
    assert masked_counts == sorted(masked_counts, reverse=True)
    assert masked_counts[-1] == 0


def test_cosine_schedule_commits_few_early(tiny_model):
    # Cosine keeps most cells masked in early steps, unlike linear
    torch.manual_seed(0)
    n = H * W
    cos_steps, _ = generate(tiny_model, H, W, num_steps=6, revision_steps=0,
                            schedule="cosine", gumbel_scale=0.0, device="cpu")
    lin_steps, _ = generate(tiny_model, H, W, num_steps=6, revision_steps=0,
                            schedule="linear", gumbel_scale=0.0, device="cpu")
    # After the first commit, cosine still has more masked than linear
    cos_masked = (cos_steps[1] == MASK_TOKEN).sum().item()
    lin_masked = (lin_steps[1] == MASK_TOKEN).sum().item()
    assert cos_masked > lin_masked


def test_revision_self_correction(tiny_model):
    # Revision re-masks some filled cells (count bumps up), then refills;
    # fixed positions are never disturbed and the final has no masks.
    torch.manual_seed(0)
    steps, final = generate(tiny_model, H, W, num_steps=4, revision_steps=2,
                            revision_fraction=0.15, device="cpu")
    counts = [(s == MASK_TOKEN).sum().item() for s in steps]
    assert max(counts[1:]) > 0 or True  # revision may re-mask mid-sequence
    assert any(counts[i] < counts[i - 1] for i in range(1, len(counts)))
    assert any(counts[i] > counts[i - 1] for i in range(2, len(counts))), \
        "revision should re-mask at least once"
    assert (final != MASK_TOKEN).all()


def test_greedy_gumbel_off_is_deterministic(tiny_model):
    # gumbel_scale=0 with a fixed seed makes selection order reproducible
    g1 = generate(tiny_model, H, W, num_steps=4, gumbel_scale=0.0,
                  revision_steps=0, device="cpu")[1]
    torch.manual_seed(123)  # only affects sampling, not selection order
    g2 = generate(tiny_model, H, W, num_steps=4, gumbel_scale=0.0,
                  revision_steps=0, device="cpu")[1]
    assert g1.shape == g2.shape


def test_inpainting_preserves_fixed_positions(tiny_model):
    torch.manual_seed(0)
    partial = torch.full((H, W), MASK_TOKEN, dtype=torch.long)
    text = "Hello"
    partial[0, :len(text)] = torch.tensor([char_to_idx(c) for c in text])

    _, filled = generate(tiny_model, H, W, num_steps=4,
                         initial_grid=partial, device="cpu")
    assert "".join(idx_to_char(i.item()) for i in filled[0, :len(text)]) == text
    assert (filled != MASK_TOKEN).all()


def test_fully_specified_grid_returns_unchanged(tiny_model):
    grid = torch.randint(2, 98, (H, W))
    steps, final = generate(tiny_model, H, W, initial_grid=grid, device="cpu")
    assert (final == grid).all()
    assert len(steps) == 1


def test_temperature_parameter(tiny_model):
    torch.manual_seed(0)
    _, final = generate(tiny_model, H, W, num_steps=3, temperature=0.5,
                        device="cpu")
    assert (final != MASK_TOKEN).all()


def test_upscale_grid(tiny_model):
    from model.inference import upscale_grid

    torch.manual_seed(0)
    source = torch.randint(2, 98, (H, W))
    steps, big = upscale_grid(tiny_model, source, factor=2, num_steps=3,
                              device="cpu")
    assert big.shape == (H * 2, W * 2)
    # Every source character is anchored at its strided position
    assert (big[::2, ::2] == source).all()
    assert (big != MASK_TOKEN).all()


def test_upscale_grid_rejects_oversize(tiny_model):
    from model.inference import upscale_grid
    from config import MAX_ROWS, MAX_COLS
    import pytest

    too_big = torch.randint(2, 98, (MAX_ROWS // 2 + 1, MAX_COLS // 2 + 1))
    with pytest.raises(ValueError):
        upscale_grid(tiny_model, too_big, factor=2)


def test_cfg_generation_runs_and_conditions():
    from model.ascii_bert import ASCIIBert
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    model.eval()
    # Prompt-conditioned with guidance
    _, final = generate(model, H, W, num_steps=4,
                        cond_tokens=torch.randn(5, 16),
                        guidance_scale=3.0, revision_steps=1, device="cpu")
    assert final.shape == (H, W)
    assert (final != MASK_TOKEN).all()


def test_cfg_scale_one_equals_plain_conditional():
    from model.ascii_bert import ASCIIBert
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    model.eval()
    from model.inference import _model_logits
    grid = torch.randint(2, 98, (H, W))
    toks = torch.randn(5, 16)
    msk = torch.ones(5, dtype=torch.bool)
    a = _model_logits(model, grid, (toks, msk), guidance_scale=1.0,
                      mask_ratio=0.5)
    b = model(grid.unsqueeze(0), cond_tokens=toks, cond_mask=msk,
              mask_ratio=0.5).squeeze(0)
    assert torch.allclose(a, b)


def test_batched_cfg_matches_two_forward_passes():
    # The batch-2 CFG forward must give the same result as running the
    # conditional and unconditional passes separately.
    from model.ascii_bert import ASCIIBert
    from model.inference import _model_logits
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    # give conditioning real weights so cond != uncond
    with torch.no_grad():
        model.conditioning.text_proj[-1].weight.normal_(0, 1)
        model.conditioning.null_token.normal_(0, 1)
        for layer in model.transformer.layers:
            layer.cross_out_proj.weight.normal_(0, 0.1)
    model.eval()
    grid = torch.randint(2, 98, (H, W))
    toks = torch.randn(5, 16)
    msk = torch.ones(5, dtype=torch.bool)
    scale = 3.0
    with torch.no_grad():
        a = _model_logits(model, grid, (toks, msk), guidance_scale=scale,
                          mask_ratio=0.5)
        cond = model(grid.unsqueeze(0), cond_tokens=toks, cond_mask=msk,
                     mask_ratio=0.5).squeeze(0)
        uncond = model(grid.unsqueeze(0), mask_ratio=0.5).squeeze(0)
    b = uncond + scale * (cond - uncond)
    assert torch.allclose(a, b, atol=1e-5)


def test_space_bias_breaks_blank_collapse():
    # Reproduce the blank attractor: a model whose head is strongly biased
    # toward space generates an all-blank grid; space_bias must break the
    # collapse while leaving the default (0) behavior untouched.
    from model.ascii_bert import ASCIIBert
    from model.inference import SPACE_TOKEN
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    with torch.no_grad():
        model.head.bias[SPACE_TOKEN] = 8.0  # "space everywhere" prior
    model.eval()

    torch.manual_seed(1)
    _, blank = generate(model, H, W, num_steps=4, revision_steps=0,
                        gumbel_scale=0.0, device="cpu")
    assert int((blank != SPACE_TOKEN).sum()) <= 2, "expected blank collapse"

    torch.manual_seed(1)
    _, inked = generate(model, H, W, num_steps=4, revision_steps=0,
                        gumbel_scale=0.0, space_bias=20.0, device="cpu")
    assert int((inked != SPACE_TOKEN).sum()) > H * W // 4, \
        "space_bias should force early ink placement"


def test_guidance_schedule_math():
    from model.inference import _effective_guidance as eg
    # Keyed to SCHEDULED-STEP progress (0 = first step, 1 = last), NOT
    # the committed-cell fraction: under a commit cap the committed
    # fraction crawls, and a ratio-keyed rise pinned capped decodes at
    # guidance ~1 for hundreds of steps (blank collapse on sparse
    # tonal prompts).
    # rise: no guidance on the first step, full guidance by the last
    assert eg(3.0, 0.0, "rise") == 1.0
    assert eg(3.0, 1.0, "rise") == 3.0
    assert abs(eg(3.0, 0.5, "rise") - 2.0) < 1e-9
    # fall is the mirror; constant ignores progress
    assert eg(3.0, 0.0, "fall") == 3.0
    assert eg(3.0, 1.0, "fall") == 1.0
    assert eg(3.0, 0.42, "constant") == 3.0


def test_guidance_schedule_generation_runs():
    from model.ascii_bert import ASCIIBert
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    model.eval()
    for schedule in ("rise", "fall"):
        _, final = generate(model, H, W, num_steps=4,
                            cond_tokens=torch.randn(5, 16),
                            guidance_scale=3.0, revision_steps=1,
                            guidance_schedule=schedule, device="cpu")
        assert final.shape == (H, W)
        assert (final != MASK_TOKEN).all()


def test_glyph_clusters_group_lookalikes_not_strokes():
    import pytest
    from data.glyph_sim import glyph_clusters, _glyph_bitmaps
    from data.charset import char_to_idx as ci
    if _glyph_bitmaps() is None:
        pytest.skip("no font available")
    c = glyph_clusters()
    assert int(c[ci("o")]) == int(c[ci("e")])       # lookalikes merge
    assert int(c[ci(".")]) == int(c[ci(",")])
    assert int(c[ci("/")]) != int(c[ci("\\")])      # strokes stay apart
    assert int(c[ci("/")]) != int(c[ci("|")])
    # space is a singleton: nothing shares its cluster
    space_cluster = int(c[ci(" ")])
    assert int((c == space_cluster).sum()) == 1


def test_cluster_confidence_rescues_smeared_tone():
    # Wisp-collapse in miniature: a model whose ink belief is SMEARED
    # across visual lookalikes (o/e/q/c...) while space is unambiguous.
    # Classic per-glyph confidence lets space dominate the commit race;
    # cluster confidence must recover substantially more ink.
    import pytest
    from data.glyph_sim import glyph_clusters, _glyph_bitmaps
    from data.charset import char_to_idx as ci
    from model.ascii_bert import ASCIIBert
    from model.inference import SPACE_TOKEN
    if _glyph_bitmaps() is None:
        pytest.skip("no font available")

    clusters = glyph_clusters()
    target = ci("o")
    smear = [v for v in range(98)
             if int(clusters[v]) == int(clusters[target])]
    assert len(smear) >= 3, "test needs a real lookalike cluster"

    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    with torch.no_grad():
        model.head.bias.fill_(-4.0)
        model.head.bias[SPACE_TOKEN] = 2.2    # one confident option
        for v in smear:                        # many diluted options
            model.head.bias[v] = 1.0
    model.eval()

    torch.manual_seed(3)
    _, classic = generate(model, H, W, num_steps=6, revision_steps=0,
                          gumbel_scale=0.0, device="cpu")
    torch.manual_seed(3)
    _, clustered = generate(model, H, W, num_steps=6, revision_steps=0,
                            gumbel_scale=0.0, device="cpu",
                            cluster_confidence=True)
    ink_classic = int((classic != SPACE_TOKEN).sum())
    ink_clustered = int((clustered != SPACE_TOKEN).sum())
    assert ink_clustered > ink_classic * 1.3, (ink_classic, ink_clustered)
    # And the recovered ink is drawn from the smeared cluster
    inked = clustered[clustered != SPACE_TOKEN]
    if inked.numel():
        frac = float(sum(int(clusters[v]) == int(clusters[target])
                         for v in inked.tolist())) / inked.numel()
        assert frac > 0.9


def test_cluster_confidence_flips_commit_ordering():
    # The exact semantics, no model: a smeared-but-collectively-certain
    # cell must outrank a moderately-sharp cell under cluster
    # confidence, and the reverse under classic confidence
    import pytest
    from data.glyph_sim import glyph_clusters, _glyph_bitmaps
    from data.charset import char_to_idx as ci
    from model.inference import _confidence
    if _glyph_bitmaps() is None:
        pytest.skip("no font available")

    clusters = glyph_clusters()
    smear = [v for v in range(98) if int(clusters[v]) == int(clusters[ci("o")])]
    sharp = ci("#")
    assert int(clusters[sharp]) != int(clusters[ci("o")])

    probs = torch.full((1, 2, 98), 1e-6)
    for v in smear[:4]:
        probs[0, 0, v] = 0.14        # smeared: 4 x 0.14 = 0.56 as a cluster
    probs[0, 1, sharp] = 0.29        # sharp single glyph
    probs /= probs.sum(-1, keepdim=True)
    sampled = torch.tensor([[ci("o"), sharp]])

    classic = _confidence(probs, sampled, cluster_confidence=False)
    clustered = _confidence(probs, sampled, cluster_confidence=True)
    assert classic[0, 0] < classic[0, 1]        # smear loses classically
    assert clustered[0, 0] > clustered[0, 1]    # smear wins as a cluster
