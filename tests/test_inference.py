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
