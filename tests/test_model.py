import torch

from config import VOCAB_SIZE
from model.ascii_bert import ASCIIBert
from model.embeddings import RoPE2D


H, W = 12, 20


def test_forward_shape(tiny_model):
    x = torch.randint(0, VOCAB_SIZE, (2, H, W))
    logits = tiny_model(x)
    assert logits.shape == (2, H, W, VOCAB_SIZE)


def test_loss_only_on_masked_positions(tiny_model):
    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    target = torch.randint(2, VOCAB_SIZE, (1, H, W))
    mask = torch.zeros(1, H, W, dtype=torch.bool)
    mask[0, 0, :5] = True

    logits = tiny_model(x)
    loss = tiny_model.compute_loss(logits, target, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)

    # Changing targets outside the mask must not change the loss
    target2 = target.clone()
    target2[~mask] = 2
    loss2 = tiny_model.compute_loss(logits, target2, mask)
    assert torch.allclose(loss, loss2)


def test_training_step_reduces_loss():
    """A tiny model overfitting a single batch should reduce its loss."""
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, VOCAB_SIZE, (4, H, W))
    target = torch.randint(2, VOCAB_SIZE, (4, H, W))
    mask = torch.rand(4, H, W) > 0.5

    losses = []
    for _ in range(15):
        logits = model(x)
        loss = model.compute_loss(logits, target, mask)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())

    assert losses[-1] < losses[0]


def test_rope_shapes_and_no_params():
    rope = RoPE2D(head_dim=16)
    assert sum(p.numel() for p in rope.parameters()) == 0
    x = torch.randn(2, 2, H * W, 16)
    out = rope(x, H, W)
    assert out.shape == x.shape
    # Rotation preserves norms
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_variable_grid_sizes(tiny_model):
    """RoPE-based model should handle any grid up to MAX_ROWS x MAX_COLS."""
    for h, w in [(8, 8), (24, 40), (48, 80)]:
        x = torch.randint(0, VOCAB_SIZE, (1, h, w))
        logits = tiny_model(x)
        assert logits.shape == (1, h, w, VOCAB_SIZE)


def test_checkpoint_round_trip(tiny_model, tmp_path):
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": tiny_model.state_dict()}, path)
    model2 = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model2.load_state_dict(ckpt["model_state_dict"])
    model2.eval()

    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    with torch.no_grad():
        assert torch.allclose(tiny_model(x), model2(x))


def test_rope_buffers_not_in_checkpoint(tiny_model):
    """RoPE tables are derived constants; grid-size config changes must
    not invalidate checkpoints."""
    assert not any("rope" in k for k in tiny_model.state_dict())


def test_rope_covers_upscaled_grids():
    from config import MAX_ROWS, MAX_COLS
    assert MAX_ROWS >= 96 and MAX_COLS >= 160
    rope = RoPE2D(head_dim=16)
    x = torch.randn(1, 2, MAX_ROWS * MAX_COLS, 16)
    out = rope(x, MAX_ROWS, MAX_COLS)  # elementwise; cheap even at max size
    assert out.shape == x.shape


def _train_conditioning(model):
    """Give the zero-init conditioning pathways non-zero weights so tests
    can verify they actually condition."""
    with torch.no_grad():
        model.conditioning.text_proj[-1].weight.normal_(0, 1)
        model.conditioning.ratio_mlp[-1].weight.normal_(0, 1)
        for layer in model.transformer.layers:
            layer.cross_out_proj.weight.normal_(0, 0.1)


def test_prompt_and_ratio_conditioning_change_output():
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    _train_conditioning(model)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    toks = torch.randn(5, 16)                        # 5 caption tokens
    with torch.no_grad():
        base = model(x)                              # null prompt, ratio 0
        cond = model(x, cond_tokens=toks)            # a real prompt
        r80 = model(x, mask_ratio=0.8)               # different noise level
        dropped = model(x, cond_tokens=toks,
                        cond_drop=torch.tensor([True]))  # forced to null
    assert not torch.allclose(base, cond), "prompt must affect output"
    assert not torch.allclose(base, r80), "mask ratio must affect output"
    assert torch.allclose(base, dropped), "cond_drop must fall back to null"


def test_cond_mask_hides_padding_tokens():
    # Padding rows behind the mask must not influence the output
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    _train_conditioning(model)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    toks = torch.randn(6, 16)
    msk = torch.tensor([True, True, True, False, False, False])
    toks2 = toks.clone()
    toks2[3:] = 999.0  # garbage in the padding positions
    with torch.no_grad():
        a = model(x, cond_tokens=toks, cond_mask=msk)
        b = model(x, cond_tokens=toks2, cond_mask=msk)
    assert torch.allclose(a, b, atol=1e-5), \
        "masked-out caption tokens must be invisible"


def test_conditioning_accepts_batched_tokens():
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    x = torch.randint(0, VOCAB_SIZE, (4, H, W))
    toks = torch.randn(4, 7, 16)
    msk = torch.rand(4, 7) > 0.3
    msk[:, 0] = True
    drop = torch.tensor([False, True, False, True])
    ratios = torch.rand(4)
    out = model(x, cond_tokens=toks, cond_mask=msk, cond_drop=drop,
                mask_ratio=ratios)
    assert out.shape == (4, H, W, VOCAB_SIZE)


def test_conditioning_overfit_separates_two_prompts():
    """End-to-end: a tiny model must learn to produce DIFFERENT outputs for
    two different caption contexts — the property the whole prompt-to-ASCII
    goal rests on. Catches dead cross-attention wiring before a paid run."""
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    model.train()
    model.transformer.gradient_checkpointing = False
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    h, w = 8, 10
    toks_a, toks_b = torch.randn(3, 16), torch.randn(3, 16)
    target_a = torch.full((1, h, w), 10)   # caption A -> all char 10
    target_b = torch.full((1, h, w), 40)   # caption B -> all char 40
    x = torch.full((1, h, w), 1)           # fully masked input
    mask = torch.ones(1, h, w, dtype=torch.bool)

    for _ in range(60):
        for toks, target in [(toks_a, target_a), (toks_b, target_b)]:
            logits = model(x, cond_tokens=toks, mask_ratio=1.0)
            loss = model.compute_loss(logits, target, mask)
            loss.backward()
            opt.step()
            opt.zero_grad()

    model.eval()
    with torch.no_grad():
        pred_a = model(x, cond_tokens=toks_a, mask_ratio=1.0).argmax(-1)
        pred_b = model(x, cond_tokens=toks_b, mask_ratio=1.0).argmax(-1)
    acc_a = (pred_a == target_a).float().mean().item()
    acc_b = (pred_b == target_b).float().mean().item()
    assert acc_a > 0.9 and acc_b > 0.9, (
        f"conditioning failed to separate prompts (acc {acc_a}, {acc_b})")


def test_soft_target_loss_matches_ce_when_onehot():
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    x = torch.randint(0, VOCAB_SIZE, (2, H, W))
    target = torch.randint(2, VOCAB_SIZE, (2, H, W))
    mask = torch.rand(2, H, W) > 0.5
    logits = model(x)

    ce = model.compute_loss(logits, target, mask)
    onehot = torch.eye(VOCAB_SIZE)
    soft = model.compute_loss(logits, target, mask, soft_target_matrix=onehot)
    assert torch.allclose(ce, soft, atol=1e-5)


def test_soft_target_loss_gives_partial_credit():
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    x = torch.randint(0, VOCAB_SIZE, (2, H, W))
    target = torch.randint(2, VOCAB_SIZE, (2, H, W))
    mask = torch.ones(2, H, W, dtype=torch.bool)
    logits = model(x)
    # A smoothed target matrix should give a different (softer) loss
    smooth = 0.9 * torch.eye(VOCAB_SIZE) + 0.1 / VOCAB_SIZE
    ce = model.compute_loss(logits, target, mask)
    soft = model.compute_loss(logits, target, mask, soft_target_matrix=smooth)
    assert not torch.allclose(ce, soft)


def test_conditioning_is_noop_at_init():
    # Zero-init conditioning: a freshly built model ignores prompt/ratio,
    # so a checkpoint trained WITHOUT conditioning loads (strict=False)
    # and behaves identically.
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    with torch.no_grad():
        a = model(x)
        b = model(x, cond_tokens=torch.randn(5, 16), mask_ratio=0.9)
    assert torch.allclose(a, b), "conditioning must be a no-op before training"


def test_pre_conditioning_checkpoint_loads():
    # Simulate an old checkpoint: state dict without the conditioning
    # pathway (ConditioningEmbedding + cross-attention). It must load via
    # load_compatible_state and report conditioned=False.
    import pytest
    from model.ascii_bert import load_compatible_state
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    state = {k: v for k, v in model.state_dict().items()
             if "conditioning" not in k and "cross" not in k}
    fresh = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    assert load_compatible_state(fresh, state) is False

    # A full state dict reports conditioned=True
    fresh2 = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                       text_dim=16)
    assert load_compatible_state(fresh2, model.state_dict()) is True

    # A genuinely mismatched checkpoint fails loudly
    broken = dict(state)
    del broken["head.weight"]
    with pytest.raises(ValueError):
        load_compatible_state(fresh, broken)
