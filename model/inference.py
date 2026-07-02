import torch
import torch.nn.functional as F

from data.charset import MASK_TOKEN


@torch.no_grad()
def generate(model, grid_h, grid_w, num_steps=10, temperature=1.0,
             initial_grid=None, device="cpu"):
    """
    Generate ASCII art via iterative unmasking (MaskGIT-style).

    Args:
        model: trained ASCIIBert model (in eval mode)
        grid_h: grid height
        grid_w: grid width
        num_steps: number of iterative unmasking steps
        temperature: sampling temperature
        initial_grid: optional [H, W] long tensor with some non-MASK chars
                      (for inpainting — those positions stay fixed)
        device: torch device

    Returns:
        steps: list of [H, W] long tensors (grid at each step)
        final: [H, W] long tensor (final result)
    """
    model.eval()

    # 1. Initialize grid
    if initial_grid is not None:
        grid = initial_grid.clone().to(device)
    else:
        grid = torch.full((grid_h, grid_w), MASK_TOKEN, dtype=torch.long, device=device)

    # 2. Identify fixed vs free positions
    fixed = (grid != MASK_TOKEN)  # [H, W] bool — True = keep as-is

    steps = [grid.clone().cpu()]

    # Count total free (masked) positions
    total_free = (~fixed).sum().item()
    if total_free == 0:
        return steps, grid.cpu()

    for step in range(num_steps):
        # Find currently masked positions
        is_masked = (grid == MASK_TOKEN)
        num_masked = is_masked.sum().item()
        if num_masked == 0:
            break

        # Forward pass
        logits = model(grid.unsqueeze(0))  # [1, H, W, V]
        logits = logits.squeeze(0)          # [H, W, V]

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        probs = F.softmax(logits, dim=-1)   # [H, W, V]

        # Confidence = max probability at each position
        confidence, _ = probs.max(dim=-1)   # [H, W]

        # Only consider currently masked positions
        confidence[~is_masked] = -1.0

        # How many to unmask this step (linear schedule)
        remaining_steps = num_steps - step
        num_to_unmask = max(1, num_masked // remaining_steps)

        # Flatten, pick top-k most confident masked positions
        flat_conf = confidence.view(-1)
        _, topk_idx = flat_conf.topk(num_to_unmask)

        # Sample from distributions at those positions
        flat_probs = probs.view(-1, probs.size(-1))  # [H*W, V]
        for idx in topk_idx:
            sampled = torch.multinomial(flat_probs[idx], 1).item()
            r, c = idx.item() // grid_w, idx.item() % grid_w
            grid[r, c] = sampled

        steps.append(grid.clone().cpu())

    # Final pass: unmask any remaining positions
    is_masked = (grid == MASK_TOKEN)
    if is_masked.any():
        logits = model(grid.unsqueeze(0)).squeeze(0)
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        masked_indices = is_masked.view(-1).nonzero(as_tuple=True)[0]
        for idx in masked_indices:
            sampled = torch.multinomial(flat_probs[idx], 1).item()
            r, c = idx.item() // grid_w, idx.item() % grid_w
            grid[r, c] = sampled
        steps.append(grid.clone().cpu())

    return steps, grid.cpu()


@torch.no_grad()
def upscale_grid(model, grid, factor=2, num_steps=10, temperature=1.0,
                 device="cpu"):
    """MaskGIT-style super-resolution: enlarge finished ASCII art.

    Anchors each source character at a strided position on a canvas
    `factor` times larger, masks everything in between, and lets the
    model inpaint the gaps. Works because 2D RoPE attention is relative:
    the model runs on any grid up to MAX_ROWS x MAX_COLS even though it
    trained at 48x80.

    Anchoring every source cell (including spaces) keeps the result
    faithful to the input; the model only ever fills gaps, it cannot
    redraw or move existing strokes.

    Args:
        model: trained ASCIIBert (eval mode)
        grid: [H, W] long tensor of character indices (no MASK tokens)
        factor: integer upscale factor
        num_steps: unmasking steps for the fill
        temperature: sampling temperature
        device: torch device

    Returns:
        steps: list of [H*factor, W*factor] grids (fill progression)
        final: [H*factor, W*factor] long tensor
    """
    from config import MAX_ROWS, MAX_COLS

    h, w = grid.shape
    big_h, big_w = h * factor, w * factor
    if big_h > MAX_ROWS or big_w > MAX_COLS:
        raise ValueError(
            f"Upscaled grid {big_h}x{big_w} exceeds RoPE precomputation "
            f"({MAX_ROWS}x{MAX_COLS}). Lower the factor or raise "
            f"MAX_ROWS/MAX_COLS in config.py.")

    canvas = torch.full((big_h, big_w), MASK_TOKEN, dtype=torch.long)
    canvas[::factor, ::factor] = grid.cpu()

    return generate(model, big_h, big_w, num_steps=num_steps,
                    temperature=temperature, initial_grid=canvas,
                    device=device)
