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
