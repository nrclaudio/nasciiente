import torch
import torch.nn as nn

from config import VOCAB_SIZE, EMBED_DIM, MAX_ROWS, MAX_COLS, NUM_HEADS, DROPOUT


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        return self.embedding(x)


class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings.

    Splits each attention head into two halves — first half encodes row
    position, second half encodes column position. Gives the attention
    mechanism relative position awareness in both spatial dimensions
    without adding any trainable parameters.
    """

    def __init__(self, head_dim, max_rows=MAX_ROWS, max_cols=MAX_COLS,
                 theta=10000.0):
        super().__init__()
        half = head_dim // 2
        # Frequency bands (shared for row and col — the position indices differ)
        freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))

        rows = torch.arange(max_rows).float()
        cols = torch.arange(max_cols).float()

        row_angles = torch.outer(rows, freqs)  # [max_rows, half//2]
        col_angles = torch.outer(cols, freqs)  # [max_cols, half//2]

        # persistent=False: these are derived constants, not weights.
        # Keeping them out of checkpoints lets MAX_ROWS/MAX_COLS change
        # without invalidating saved models.
        self.register_buffer("row_cos", row_angles.cos(), persistent=False)
        self.register_buffer("row_sin", row_angles.sin(), persistent=False)
        self.register_buffer("col_cos", col_angles.cos(), persistent=False)
        self.register_buffer("col_sin", col_angles.sin(), persistent=False)

    def forward(self, x, h, w):
        """Apply 2D RoPE to query or key tensor.

        Args:
            x: [B, num_heads, H*W, head_dim]
            h, w: grid dimensions (H, W)
        Returns:
            Rotated tensor with same shape.
        """
        # Position indices for the flattened grid
        row_idx = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w).reshape(-1)
        col_idx = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w).reshape(-1)

        rc = self.row_cos[row_idx]  # [H*W, half//2]
        rs = self.row_sin[row_idx]
        cc = self.col_cos[col_idx]
        cs = self.col_sin[col_idx]

        half = x.shape[-1] // 2
        x_row = x[..., :half]
        x_col = x[..., half:]

        x_row = _apply_rope(x_row, rc, rs)
        x_col = _apply_rope(x_col, cc, cs)

        return torch.cat([x_row, x_col], dim=-1)


def _apply_rope(x, cos, sin):
    """Rotate interleaved pairs of dimensions.

    Args:
        x: [B, nh, seq, d]
        cos, sin: [seq, d//2]
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d//2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)


class ConditioningEmbedding(nn.Module):
    """Global conditioning: caption tokens + current mask ratio.

    The prompt arrives as the frozen text encoder's per-token hidden
    states (see data/text_embed.py). This module produces two things:

    - a global [B, D] vector added to every token embedding (Muse-style):
      the masked mean of the caption tokens projected into the stream,
      plus a mask-ratio embedding — the denoising "noise level" every
      diffusion model gets but a plain masked LM does not;
    - the caption-token context [B, L, text_dim] + validity mask that the
      transformer blocks cross-attend to for compositional control.

    A learned null token stands in for "no prompt" — used for uncaptioned
    data and as the unconditional branch of classifier-free guidance.
    """

    def __init__(self, text_dim, embed_dim):
        super().__init__()
        self.text_dim = text_dim
        self.null_token = nn.Parameter(torch.zeros(text_dim))
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.ratio_mlp = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Zero-init the output layers so conditioning is a no-op until
        # trained. This makes the module identity at start (like AdaLN-Zero)
        # AND lets a checkpoint trained without conditioning load with
        # strict=False and behave exactly as before — the added cond vector
        # is zero (the cross-attention blocks zero-init their own output).
        nn.init.zeros_(self.text_proj[-1].weight)
        nn.init.zeros_(self.text_proj[-1].bias)
        nn.init.zeros_(self.ratio_mlp[-1].weight)
        nn.init.zeros_(self.ratio_mlp[-1].bias)

    def forward(self, batch, device, cond_tokens=None, cond_mask=None,
                cond_drop=None, mask_ratio=None):
        """
        Args:
            batch: batch size B
            device: torch device
            cond_tokens: None, [L, text_dim], or [B, L, text_dim] float —
                         caption-token embeddings (None -> null token)
            cond_mask: optional [L] or [B, L] bool, True = real token.
                       None -> all tokens valid.
            cond_drop: optional [B] bool — True rows use the null token
                       instead of their caption (training-time CFG dropout /
                       the unconditional half of guidance). Rows whose mask
                       has no valid token are always treated as dropped.
            mask_ratio: None, float, or [B] float tensor (None -> 0.0)
        Returns:
            (cond_vec [B, embed_dim], ctx [B, L, text_dim], ctx_mask [B, L])
        """
        null = self.null_token
        if cond_tokens is None:
            ctx = null.view(1, 1, -1).expand(batch, 1, -1)
            ctx_mask = torch.ones(batch, 1, dtype=torch.bool, device=device)
        else:
            ctx = cond_tokens.to(device=device, dtype=null.dtype)
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0)
            ctx = ctx.expand(batch, -1, -1)
            L = ctx.shape[1]
            if cond_mask is None:
                ctx_mask = torch.ones(batch, L, dtype=torch.bool,
                                      device=device)
            else:
                ctx_mask = cond_mask.to(device).bool()
                if ctx_mask.dim() == 1:
                    ctx_mask = ctx_mask.unsqueeze(0)
                ctx_mask = ctx_mask.expand(batch, -1)

            # Null rows: explicit drops plus rows with no valid token
            # (uncaptioned samples) — they get the null token at position 0.
            # Applied unconditionally (drop may be all-False): torch.where
            # keeps null_token in the autograd graph every step, which DDP
            # requires (a param that only sometimes gets gradients trips
            # its reducer), and skipping a .any() check avoids a GPU sync.
            drop = ~ctx_mask.any(dim=1)
            if cond_drop is not None:
                drop = drop | cond_drop.to(device).bool()
            ctx = torch.where(drop.view(-1, 1, 1), null.expand_as(ctx), ctx)
            first_only = torch.zeros_like(ctx_mask)
            first_only[:, 0] = True
            ctx_mask = torch.where(drop.view(-1, 1), first_only, ctx_mask)

        # Global vector: masked mean of the context tokens
        denom = ctx_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (ctx * ctx_mask.unsqueeze(-1)).sum(dim=1) / denom
        cond_vec = self.text_proj(pooled)  # [B, D]

        if mask_ratio is None:
            ratio = torch.zeros(batch, 1, device=device)
        elif torch.is_tensor(mask_ratio):
            ratio = mask_ratio.to(device).float().view(batch, 1)
        else:
            ratio = torch.full((batch, 1), float(mask_ratio), device=device)
        return cond_vec + self.ratio_mlp(ratio), ctx, ctx_mask


class CombinedEmbedding(nn.Module):
    """Token embedding with layer norm and dropout, plus optional global
    conditioning added to every position.

    Positional information is handled by RoPE2D inside the attention layers,
    so this module only provides token embeddings.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids, cond=None):
        """
        Args:
            token_ids: [B, H, W] integer tensor
            cond: optional [B, embed_dim] conditioning vector, added to
                  every token before norm
        Returns:
            [B, H*W, embed_dim] float tensor
        """
        B, H, W = token_ids.shape
        tok = self.token_embed(token_ids)              # [B, H, W, D]
        x = tok.view(B, H * W, -1)                     # [B, H*W, D]
        if cond is not None:
            x = x + cond.unsqueeze(1)                  # broadcast over tokens
        x = self.norm(x)
        x = self.dropout(x)
        return x
