import math

import torch
import torch.nn.functional as F

from config import DECODE_CAP_BELOW
from data.charset import MASK_TOKEN, char_to_idx

SPACE_TOKEN = char_to_idx(" ")


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


_CLUSTERS = {}


def _cluster_ids(device, threshold=0.80):
    """Cached [V] visual-cluster ids for cluster-confidence decoding."""
    key = (str(device), threshold)
    if key not in _CLUSTERS:
        from data.glyph_sim import glyph_clusters
        _CLUSTERS[key] = glyph_clusters(threshold).to(device)
    return _CLUSTERS[key]


def _confidence(probs, sampled, cluster_confidence, threshold=0.80):
    """Per-cell commitment confidence for the sampled glyphs.

    Classic MaskGIT uses P(sampled glyph). But when training labels are
    ambiguous between visual lookalikes (o/e/q for the same tone), a
    well-trained model's probability is a SMEAR across the cluster —
    individually low, collectively high — so tone cells always lose the
    confidence race to unambiguous cells and never commit (wisp
    collapse). Cluster confidence ranks by the total mass of the
    sampled glyph's visual cluster: "how sure am I that something LIKE
    this goes here", which is the question commitment actually needs
    answered. The sampled glyph itself is unchanged — full vocabulary
    expressiveness is preserved.
    """
    p_sampled = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    if not cluster_confidence:
        return torch.log(p_sampled.clamp_min(1e-9))
    cid = _cluster_ids(probs.device, threshold)          # [V]
    n_clusters = int(cid.max()) + 1
    mass = torch.zeros(*probs.shape[:-1], n_clusters,
                       device=probs.device, dtype=probs.dtype)
    mass.scatter_add_(-1, cid.expand_as(probs), probs)
    sampled_cluster = cid[sampled]
    conf = mass.gather(-1, sampled_cluster.unsqueeze(-1)).squeeze(-1)
    return torch.log(conf.clamp_min(1e-9))


_HALTON_CACHE = {}


def _halton_ranks(H, W, device):
    """Rank of every grid cell in a 2D Halton (base 2/3) low-discrepancy
    visit order — lower rank = committed earlier.

    Confidence-ordered unmasking commits spatially clustered, highly
    correlated cells together (low information gain per step; the formal
    version of the edge-echo and space-first-cascade pathologies — arXiv
    2503.17076). A low-discrepancy order spreads every step's commits
    uniformly across the canvas instead: each committed cell pins its
    neighborhood, and no step ever commits a whole edge's worth of
    correlated hypotheses at once. Training-free.
    """
    key = (H, W, str(device))
    if key not in _HALTON_CACHE:
        def halton(i, base):
            f, r = 1.0, 0.0
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        ranks = torch.full((H * W,), -1, dtype=torch.long)
        rank, i = 0, 1
        while rank < H * W and i <= 200 * H * W:
            r = int(halton(i, 2) * H)
            c = int(halton(i, 3) * W)
            i += 1
            idx = min(r, H - 1) * W + min(c, W - 1)
            if ranks[idx] < 0:
                ranks[idx] = rank
                rank += 1
        for idx in (ranks < 0).nonzero().flatten().tolist():
            ranks[idx] = rank  # stragglers (if any) go last, scan order
            rank += 1
        _HALTON_CACHE[key] = ranks.view(H, W).to(device)
    return _HALTON_CACHE[key]


def _effective_guidance(scale, progress, schedule):
    """Per-step CFG scale under a schedule over the decode.

    "constant": classic CFG. "rise": start at 1 on the empty canvas and
    grow to `scale` across the decode (Muse's increasing schedule) — the
    empty-canvas CFG direction is essentially "add ink everywhere", so a
    full-strength scale floods early commits; details still sharpen
    under full guidance late. "fall" is the mirror, for A/B probes.

    `progress` is SCHEDULED-STEP progress (step/num_steps, clamped to
    1), NOT the committed-cell fraction. Uncapped they track each other
    (the cosine schedule ties commits to steps), but under a commit cap
    the committed fraction crawls — keyed to it, a capped decode spent
    hundreds of steps at guidance ~1, where the blank attractor rules,
    and sparse tonal prompts ("a dragon") collapsed to near-empty grids
    while the same checkpoint drew them fine uncapped. Keying the ramp
    to planned steps restores full guidance while most cells are still
    open.
    """
    if schedule == "rise":
        return 1.0 + (scale - 1.0) * progress
    if schedule == "fall":
        return 1.0 + (scale - 1.0) * (1.0 - progress)
    return scale


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


def _model_logits(model, grid, cond, guidance_scale, mask_ratio):
    """Model logits for one grid, with classifier-free guidance.

    cond is None or a (cond_tokens [L, D], cond_mask [L]) pair. CFG: push
    the conditional prediction away from the unconditional one, logits =
    uncond + scale * (cond - uncond). scale=1 or no prompt means a single
    plain forward pass; with guidance the conditional and unconditional
    predictions run as ONE batch-2 forward (same math as two passes at
    roughly half the latency).
    """
    if cond is None or guidance_scale == 1.0:
        kwargs = ({} if cond is None
                  else dict(cond_tokens=cond[0], cond_mask=cond[1]))
        return model(grid.unsqueeze(0), mask_ratio=mask_ratio,
                     **kwargs).squeeze(0)
    batch = grid.unsqueeze(0).expand(2, -1, -1)
    drop = torch.tensor([False, True], device=grid.device)
    ratios = torch.full((2,), float(mask_ratio), device=grid.device)
    logits = model(batch, cond_tokens=cond[0], cond_mask=cond[1],
                   cond_drop=drop, mask_ratio=ratios)
    cond_l, uncond_l = logits[0], logits[1]
    return uncond_l + guidance_scale * (cond_l - uncond_l)


def _critic_scores(model, grid, cond, mask_ratio):
    """Per-cell critic logits ("is this glyph correct?") for the grid as
    it stands. Single conditional forward — no CFG extrapolation; the
    critic is a calibrated judge, not a direction to amplify."""
    kwargs = ({} if cond is None
              else dict(cond_tokens=cond[0], cond_mask=cond[1]))
    _, critic = model(grid.unsqueeze(0), mask_ratio=mask_ratio,
                      return_critic=True, **kwargs)
    return critic.squeeze(0)


def _iterative_fill(model, grid, num_steps, temperature, schedule,
                    gumbel_scale, steps_out,
                    cond=None, guidance_scale=1.0, space_bias=0.0,
                    guidance_schedule="constant",
                    cluster_confidence=False, max_commit=None,
                    cap_below=1.0, order="confidence",
                    critic_confidence=False, remask_eta=0.0,
                    probs_out=None):
    """Fill every currently-[MASK] cell of `grid` in place (MaskGIT-style).

    Cells that are not [MASK] on entry are never touched, so fixed/anchor
    positions and already-committed cells are preserved automatically.

    Selection each step: sample a candidate for every masked cell, then
    commit only the top-k by confidence, where confidence = log P(sampled)
    plus Gumbel noise annealed to zero across steps (explore early, commit
    late). k follows the cosine (or linear) schedule. The model is told the
    current mask ratio each step (its denoising noise level).

    space_bias > 0 counteracts the blank-grid attractor: on dense/uncertain
    targets the per-cell argmax is "space" almost everywhere at high mask
    ratios, so confidence-ordered commits lay a sea of spaces first and the
    blank context locks in. The bias is subtracted from the space logit
    scaled by the CURRENT mask ratio — full strength on an empty canvas,
    zero once the grid is mostly committed — forcing early ink placement
    while leaving late-stage space placement free.

    max_commit caps how many cells may commit per step. Without it the
    cosine schedule commits the MOST cells in the LAST steps — and the
    low-confidence cells left at the end are precisely the ambiguous
    ones (e.g. every column where a shape's unseen edge could fall), so
    they mass-commit in one parallel step, each independently sampling
    a different answer: several offset edges — the "edge echo". The cap
    resolves that tail incrementally, each commit becoming context for
    the next decision, at the price of extra forward passes (the loop
    keeps running past num_steps until the grid is full).

    cap_below gates WHEN the cap binds: only once the mask ratio drops
    to/under this threshold (1.0 = from the first step). Capping the
    whole decode starved sparse tonal subjects into blank collapse —
    the head became a greedy space-cascade, and both exploration noise
    and the rise schedule expired at num_steps, 6% into a ~480-step
    capped decode ("a dragon": 8 ink capped vs 865 uncapped, SAME
    checkpoint). Echo never lives in the head (confident cells commit
    coherently there — see the inpaint probes); it is born in the
    mass-committed ambiguous tail. So: head uncapped (layout forms
    under exploration + rising guidance, exactly the pre-cap decode),
    tail capped (ambiguity resolved sequentially).
    """
    n0 = int((grid == MASK_TOKEN).sum().item())
    if n0 == 0:
        return

    # Cells this fill owns; anchors/fixed context are never remasked
    refillable = grid == MASK_TOKEN

    total = grid.numel()
    step = 0
    while True:
        is_masked = (grid == MASK_TOKEN)
        num_masked = int(is_masked.sum().item())
        if num_masked == 0:
            break

        mask_ratio = num_masked / total
        progress = min(1.0, (step + 1) / max(1, num_steps))
        g = _effective_guidance(guidance_scale, progress,
                                guidance_schedule)
        logits = _model_logits(model, grid, cond, g,
                               mask_ratio)                # [H, W, V]
        if temperature != 1.0:
            logits = logits / temperature
        if space_bias > 0:
            logits[..., SPACE_TOKEN] -= space_bias * mask_ratio
        probs = F.softmax(logits, dim=-1)

        # Probability-cloud capture: pre-commit grid + full per-cell
        # distribution, for rendering the decode's uncertainty
        # (fp16 keeps a long capped tail affordable)
        if probs_out is not None:
            probs_out.append((grid.clone().cpu(), probs.half().cpu()))

        sampled = _sample_all(probs)                     # [H, W]

        # Commit-order ranking. Three sources, in precedence order:
        #   halton — position-based low-discrepancy spread (arXiv
        #     2503.17076): ignores confidence entirely, so it neither
        #     clusters correlated commits (echo) nor greedily commits
        #     the safest cells first (space cascade).
        #   critic — trained Token-Critic judges the SAMPLED candidates
        #     in place (one extra forward on the tentatively-filled
        #     grid), replacing generator confidence.
        #   confidence — log P(sampled), the MaskGIT default.
        if order == "halton":
            conf = -_halton_ranks(*grid.shape, grid.device).float()
        elif critic_confidence:
            tentative = torch.where(is_masked, sampled, grid)
            conf = _critic_scores(model, tentative, cond, mask_ratio)
        else:
            conf = _confidence(probs, sampled, cluster_confidence)
            anneal = gumbel_scale * max(0.0, 1.0 - step / max(1, num_steps))
            if anneal > 0:
                conf = conf + anneal * _gumbel_like(conf)
        conf = conf.clone()
        conf[~is_masked] = float("-inf")

        # Commit enough to reach the schedule's masked target for this
        # step (target 0 past num_steps — only reachable with a cap)
        target = _num_masked_target(n0, min(step + 1, num_steps),
                                    num_steps, schedule)
        num_commit = num_masked - target
        num_commit = max(1, min(num_commit, num_masked))
        if max_commit is not None and mask_ratio <= cap_below:
            num_commit = min(num_commit, max_commit)

        commit_idx = conf.view(-1).topk(num_commit).indices
        grid.view(-1)[commit_idx] = sampled.view(-1)[commit_idx]

        # ReMDM-style in-loop remasking (arXiv 2503.00307, inference-
        # only): each step, re-mask the least-confident committed cells
        # so no commitment is final until the schedule closes. The
        # fraction anneals with scheduled-step progress (a cap/loop
        # hybrid of the paper's eta schedules), reaching 0 at num_steps
        # — remasks stay strictly below commits, so the fill always
        # terminates. This is the principled, continuous version of the
        # bolted-on revision passes (which re-roll big batches at the
        # very end instead).
        if remask_eta > 0:
            frac = remask_eta * max(0.0, 1.0 - step / max(1, num_steps))
            k = min(int(frac * num_commit), num_commit - 1)
            if k > 0:
                committed = refillable & (grid != MASK_TOKEN)
                cur_conf = _confidence(probs, grid, cluster_confidence)
                cur_conf = cur_conf.clone()
                cur_conf[~committed] = float("inf")
                worst = cur_conf.view(-1).topk(k, largest=False).indices
                grid.view(-1)[worst] = MASK_TOKEN

        steps_out.append(grid.clone().cpu())
        step += 1


@torch.no_grad()
def generate(model, grid_h, grid_w, num_steps=10, temperature=1.0,
             initial_grid=None, device="cpu", schedule="cosine",
             gumbel_scale=1.0, revision_steps=2, revision_fraction=0.1,
             prompt=None, cond_tokens=None, cond_mask=None,
             guidance_scale=1.0, space_bias=0.0,
             guidance_schedule="constant", cluster_confidence=False,
             max_commit=None, cap_below=DECODE_CAP_BELOW,
             order="confidence", critic_confidence=False,
             remask_eta=0.0, probs_out=None):
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
        cond_tokens: optional precomputed [L, TEXT_EMB_DIM] caption-token
                     embeddings (takes precedence over `prompt`; lets
                     callers cache embeddings and avoid the encoder)
        cond_mask: optional [L] bool validity mask for cond_tokens
                   (None -> all valid)
        guidance_scale: classifier-free guidance strength (1.0 = off). Higher
                        sharpens adherence to the prompt at some diversity
                        cost.
        space_bias: anti-blank pressure (0 = off). Subtracted from the space
                    logit scaled by the current mask ratio, so an empty
                    canvas is pushed to place ink but a mostly-filled grid
                    is not. Useful when free generation collapses to blank
                    on dense/high-uncertainty checkpoints; try 2-6.
        guidance_schedule: "constant" (classic CFG), "rise" (scale grows
                    from 1 to guidance_scale as the grid commits —
                    prevents the early-decode ink flood seen at scales
                    >= 2), or "fall" (the mirror, for A/B probing).
        cluster_confidence: rank commitment by the probability mass of
                    the sampled glyph's VISUAL cluster instead of the
                    single glyph. Counters wisp-collapse on tonal
                    checkpoints (ambiguous lookalike labels smear the
                    per-glyph probabilities) while keeping the full
                    glyph vocabulary in the output.
        max_commit: cap on cells committed per step (None = schedule
                    only). Counters mode-mixing on ambiguous regions:
                    the cosine schedule mass-commits the low-confidence
                    tail in its final steps, sampling incompatible
                    hypotheses in parallel (the edge-echo mechanism).
                    A cap resolves them sequentially; the fill loops
                    past num_steps until complete.
        cap_below:  mask-ratio threshold at/below which max_commit
                    binds (default from config, ~0.35; 1.0 = whole
                    decode). Head runs uncapped so layout forms under
                    exploration and rising guidance — a fully-capped
                    decode greedy-cascades sparse tonal prompts into
                    blank; echo is born only in the ambiguous tail.
        order:      commit-order source: "confidence" (MaskGIT default,
                    optionally gumbel-noised) or "halton" (training-free
                    low-discrepancy positional spread, arXiv 2503.17076
                    — commits are spatially uniform every step, attacking
                    both clustered-commit echo and the greedy space-first
                    cascade at once).
        critic_confidence: rank commits and revisions by the trained
                    Token-Critic head instead of generator confidence
                    (requires a checkpoint trained with
                    CRITIC_LOSS_WEIGHT > 0; zero-init critics score all
                    cells 0.5 = random order). One extra forward per
                    step. Ignored when order="halton".
        remask_eta: ReMDM-style in-loop remasking intensity (0 = off,
                    try 0.1-0.5). Each step re-masks the least-confident
                    committed cells at a fraction that anneals to zero by
                    num_steps, so early commitments stay revisable while
                    layout forms (arXiv 2503.00307 — inference-only, no
                    retraining). The principled replacement for
                    revision_steps; prefer one or the other, not both.
        probs_out:  optional list; each decode step appends a tuple of
                    (pre-commit grid, [H, W, V] fp16 softmax probs) —
                    the raw material for visualizing the decode's
                    uncertainty (see /api/cloud in the app).

    Returns:
        steps: list of [H, W] long tensors (grid at each step)
        final: [H, W] long tensor (final result)
    """
    model.eval()

    if cond_tokens is None and prompt:
        from data.text_embed import embed_captions
        toks, msk = embed_captions([prompt])
        cond_tokens, cond_mask = toks[0], msk[0]
    cond = None
    if cond_tokens is not None:
        cond_tokens = torch.as_tensor(cond_tokens,
                                      dtype=torch.float).to(device)
        if cond_mask is None:
            cond_mask = torch.ones(cond_tokens.shape[0], dtype=torch.bool)
        cond = (cond_tokens, cond_mask.to(device))

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
                    gumbel_scale, steps, cond, guidance_scale, space_bias,
                    guidance_schedule, cluster_confidence, max_commit,
                    cap_below, order, critic_confidence, remask_eta,
                    probs_out=probs_out)

    # --- Revision passes (poor-man's Token-Critic self-correction) ---
    free = ~fixed
    num_free = int(free.sum().item())
    total = grid.numel()
    for _ in range(revision_steps):
        k = max(1, int(revision_fraction * num_free))

        ratio_now = int((grid == MASK_TOKEN).sum().item()) / total
        if critic_confidence:
            # Trained judge of committed glyphs — exactly its training
            # distribution (visible cells, "is this correct?")
            cur_conf = _critic_scores(model, grid, cond, ratio_now)
        else:
            logits = _model_logits(model, grid, cond, guidance_scale,
                                   ratio_now)
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            cur_conf = _confidence(probs, grid, cluster_confidence)
        cur_conf = cur_conf.clone()
        cur_conf[fixed] = float("inf")  # fixed cells are never re-masked

        # Re-mask the k least-confident free cells, then refill them
        worst = cur_conf.view(-1).topk(k, largest=False).indices
        grid.view(-1)[worst] = MASK_TOKEN
        steps.append(grid.clone().cpu())

        refill_steps = max(2, num_steps // 4)
        _iterative_fill(model, grid, refill_steps, temperature, schedule,
                        gumbel_scale, steps, cond, guidance_scale,
                        space_bias, guidance_schedule, cluster_confidence,
                        max_commit, cap_below, order, critic_confidence,
                        remask_eta, probs_out=probs_out)

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
