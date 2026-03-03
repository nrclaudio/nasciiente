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

        self.register_buffer("row_cos", row_angles.cos())
        self.register_buffer("row_sin", row_angles.sin())
        self.register_buffer("col_cos", col_angles.cos())
        self.register_buffer("col_sin", col_angles.sin())

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


class CombinedEmbedding(nn.Module):
    """Token embedding with layer norm and dropout.

    Positional information is handled by RoPE2D inside the attention layers,
    so this module only provides token embeddings.
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, dropout=DROPOUT):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        """
        Args:
            token_ids: [B, H, W] integer tensor
        Returns:
            [B, H*W, embed_dim] float tensor
        """
        B, H, W = token_ids.shape
        tok = self.token_embed(token_ids)              # [B, H, W, D]
        x = tok.view(B, H * W, -1)                     # [B, H*W, D]
        x = self.norm(x)
        x = self.dropout(x)
        return x
