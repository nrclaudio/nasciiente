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


def test_generate_monotonic_unmasking(tiny_model):
    torch.manual_seed(0)
    steps, _ = generate(tiny_model, H, W, num_steps=5, device="cpu")
    masked_counts = [(s == MASK_TOKEN).sum().item() for s in steps]
    assert masked_counts == sorted(masked_counts, reverse=True)
    assert masked_counts[-1] == 0


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
