import math

import torch
import torch.nn.functional as F

from data.charset import MASK_TOKEN


def _sample_all(probs):
    """Vectorized categorical sample over a [H, W, V] probability grid.

    Returns [H, W] long tensor of sampled token indices — one batched
    multinomial call instead of a Python loop over cells.
    """
    h, w, v = probs.shape
    flat = probs.reshape(-1, v)
    sampled = torch.multinomial(flat, 1).squeeze(-1)  # [H*W]
    return sampled.view(h, w)


def _gumbel_like(x):
    """Standard Gumbel(0,1) noise shaped like x."""
    u = torch.rand_like(x).clamp_(1e-9, 1.0)
    return -torch.log(-torch.log(u))


def _num_masked_target(total, step, num_steps, schedule):
    """How many cells should REMAIN masked after `step` (1-indexed).

    cosine (MaskGIT): cos(pi/2 * step/T) — near-flat early (few, careful
    commitments), steep late (accelerating). linear reproduces the old
    even-per-step behavior. Both reach 0 at the final step.
    """
    r = step / num_steps
    if schedule == "linear":
        frac = 1.0 - r
    else:  # cosine
        frac = math.cos(0.5 * math.pi * r)
    return int(math.floor(total * frac))


def _model_logits(model, grid, cond_emb, guidance_scale, mask_ratio):
    """Model logits for one grid, with classifier-free guidance.

    CFG: push the conditional prediction away from the unconditional one,
    logits = uncond + scale * (cond - uncond). scale=1 or no prompt means a
    single plain forward pass; with guidance the conditional and
    unconditional predictions run as ONE batch-2 forward (same math as two
    passes at roughly half the latency).
    """
    if cond_emb is None or guidance_scale == 1.0:
        return model(grid.unsqueeze(0), cond_emb=cond_emb,
                     mask_ratio=mask_ratio).squeeze(0)
    batch = grid.unsqueeze(0).expand(2, -1, -1)
    drop = torch.tensor([False, True], device=grid.device)
    ratios = torch.full((2,), float(mask_ratio), device=grid.device)
    logits = model(batch, cond_emb=cond_emb, cond_drop=drop,
                   mask_ratio=ratios)
    cond, uncond = logits[0], logits[1]
    return uncond + guidance_scale * (cond - uncond)


def _iterative_fill(model, grid, num_steps, temperature, schedule,
                    gumbel_scale, steps_out,
                    cond_emb=None, guidance_scale=1.0):
    """Fill every currently-[MASK] cell of `grid` in place (MaskGIT-style).

    Cells that are not [MASK] on entry are never touched, so fixed/anchor
    positions and already-committed cells are preserved automatically.

    Selection each step: sample a candidate for every masked cell, then
    commit only the top-k by confidence, where confidence = log P(sampled)
    plus Gumbel noise annealed to zero across steps (explore early, commit
    late). k follows the cosine (or linear) schedule. The model is told the
    current mask ratio each step (its denoising noise level).
    """
    n0 = int((grid == MASK_TOKEN).sum().item())
    if n0 == 0:
        return

    total = grid.numel()
    for step in range(num_steps):
        is_masked = (grid == MASK_TOKEN)
        num_masked = int(is_masked.sum().item())
        if num_masked == 0:
            break

        mask_ratio = num_masked / total
        logits = _model_logits(model, grid, cond_emb, guidance_scale,
                               mask_ratio)                # [H, W, V]
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        sampled = _sample_all(probs)                     # [H, W]
        sampled_prob = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # Confidence with annealed Gumbel noise (0 on the last step)
        conf = torch.log(sampled_prob.clamp_min(1e-9))
        anneal = gumbel_scale * (1.0 - step / max(1, num_steps))
        if anneal > 0:
            conf = conf + anneal * _gumbel_like(conf)
        conf[~is_masked] = float("-inf")

        # Commit enough to reach the schedule's masked target for this step
        target = _num_masked_target(n0, step + 1, num_steps, schedule)
        num_commit = num_masked - target
        num_commit = max(1, min(num_commit, num_masked))

        commit_idx = conf.view(-1).topk(num_commit).indices
        grid.view(-1)[commit_idx] = sampled.view(-1)[commit_idx]
        steps_out.append(grid.clone().cpu())

    # Safety: force-fill anything left (schedule should already reach 0)
    is_masked = (grid == MASK_TOKEN)
    if is_masked.any():
        mask_ratio = int(is_masked.sum().item()) / total
        logits = _model_logits(model, grid, cond_emb, guidance_scale,
                               mask_ratio)
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        sampled = _sample_all(probs)
        grid[is_masked] = sampled[is_masked]
        steps_out.append(grid.clone().cpu())


@torch.no_grad()
def generate(model, grid_h, grid_w, num_steps=10, temperature=1.0,
             initial_grid=None, device="cpu", schedule="cosine",
             gumbel_scale=1.0, revision_steps=2, revision_fraction=0.1,
             prompt=None, cond_emb=None, guidance_scale=1.0):
    """
    Generate ASCII art via iterative unmasking (MaskGIT-style).

    Args:
        model: trained ASCIIBert model (in eval mode)
        grid_h, grid_w: grid dimensions
        num_steps: number of iterative unmasking steps
        temperature: sampling temperature
        initial_grid: optional [H, W] long tensor with some non-MASK chars
                      (for inpainting — those positions stay fixed)
        device: torch device
        schedule: "cosine" (MaskGIT, default) or "linear"
        gumbel_scale: strength of annealed Gumbel exploration noise on the
                      confidence ranking (0 = greedy, deterministic order)
        revision_steps: rounds of self-correction after the main fill —
                        re-mask the least-confident cells and refill them
                        (0 disables; the main loop can't fix early mistakes)
        revision_fraction: fraction of free cells re-masked per revision round
        prompt: optional text prompt — embedded with the frozen text encoder
                (needs the 'transformers' package at generation time)
        cond_emb: optional precomputed [TEXT_EMB_DIM] prompt embedding
                  (takes precedence over `prompt`; lets callers cache
                  embeddings and avoid loading the encoder)
        guidance_scale: classifier-free guidance strength (1.0 = off). Higher
                        sharpens adherence to the prompt at some diversity
                        cost.

    Returns:
        steps: list of [H, W] long tensors (grid at each step)
        final: [H, W] long tensor (final result)
    """
    model.eval()

    if cond_emb is None and prompt:
        from data.text_embed import embed_texts
        cond_emb = embed_texts([prompt])[0]
    if cond_emb is not None:
        cond_emb = torch.as_tensor(cond_emb, dtype=torch.float).to(device)

    if initial_grid is not None:
        grid = initial_grid.clone().to(device)
    else:
        grid = torch.full((grid_h, grid_w), MASK_TOKEN, dtype=torch.long,
                          device=device)

    fixed = (grid != MASK_TOKEN)  # True = keep as-is, never overwrite
    steps = [grid.clone().cpu()]

    if (~fixed).sum().item() == 0:
        return steps, grid.cpu()

    # --- Main fill ---
    _iterative_fill(model, grid, num_steps, temperature, schedule,
                    gumbel_scale, steps, cond_emb, guidance_scale)

    # --- Revision passes (poor-man's Token-Critic self-correction) ---
    free = ~fixed
    num_free = int(free.sum().item())
    total = grid.numel()
    for _ in range(revision_steps):
        k = max(1, int(revision_fraction * num_free))

        logits = _model_logits(model, grid, cond_emb, guidance_scale,
                               int((grid == MASK_TOKEN).sum().item()) / total)
        if temperature != 1.0:
            logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        cur_conf = probs.gather(-1, grid.unsqueeze(-1)).squeeze(-1)  # [H, W]
        cur_conf[fixed] = float("inf")  # fixed cells are never re-masked

        # Re-mask the k least-confident free cells, then refill them
        worst = cur_conf.view(-1).topk(k, largest=False).indices
        grid.view(-1)[worst] = MASK_TOKEN
        steps.append(grid.clone().cpu())

        refill_steps = max(2, num_steps // 4)
        _iterative_fill(model, grid, refill_steps, temperature, schedule,
                        gumbel_scale, steps, cond_emb, guidance_scale)

    return steps, grid.cpu()


@torch.no_grad()
def upscale_grid(model, grid, factor=2, num_steps=10, temperature=1.0,
                 device="cpu", **gen_kwargs):
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
        gen_kwargs: forwarded to generate() (schedule, gumbel_scale, ...)

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
                    device=device, **gen_kwargs)
