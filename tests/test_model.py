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


def test_prompt_and_ratio_conditioning_change_output():
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    # Conditioning is zero-init (no-op until trained); simulate training by
    # giving it non-zero weights, then verify it actually conditions.
    with torch.no_grad():
        model.conditioning.text_proj[-1].weight.normal_(0, 1)
        model.conditioning.ratio_mlp[-1].weight.normal_(0, 1)
    model.eval()
    x = torch.randint(0, VOCAB_SIZE, (1, H, W))
    emb = torch.randn(16)
    with torch.no_grad():
        base = model(x)                              # null prompt, ratio 0
        cond = model(x, cond_emb=emb)                # a real prompt
        r80 = model(x, mask_ratio=0.8)               # different noise level
        dropped = model(x, cond_emb=emb,
                        cond_drop=torch.tensor([True]))  # forced to null
    assert not torch.allclose(base, cond), "prompt must affect output"
    assert not torch.allclose(base, r80), "mask ratio must affect output"
    assert torch.allclose(base, dropped), "cond_drop must fall back to null"


def test_conditioning_accepts_batched_embeddings():
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    x = torch.randint(0, VOCAB_SIZE, (4, H, W))
    embs = torch.randn(4, 16)
    drop = torch.tensor([False, True, False, True])
    ratios = torch.rand(4)
    out = model(x, cond_emb=embs, cond_drop=drop, mask_ratio=ratios)
    assert out.shape == (4, H, W, VOCAB_SIZE)


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
        b = model(x, cond_emb=torch.randn(16), mask_ratio=0.9)
    assert torch.allclose(a, b), "conditioning must be a no-op before training"


def test_pre_conditioning_checkpoint_loads_strict_false():
    # Simulate a run #1 checkpoint: state dict with the conditioning keys
    # removed. It must load with strict=False.
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    state = {k: v for k, v in model.state_dict().items()
             if "conditioning" not in k}
    fresh = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64,
                      text_dim=16)
    missing, unexpected = fresh.load_state_dict(state, strict=False)
    assert unexpected == []
    assert all("conditioning" in k for k in missing)
