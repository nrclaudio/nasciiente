import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import EMBED_DIM, NUM_HEADS, FFN_DIM, DROPOUT, NUM_LAYERS, MAX_ROWS, MAX_COLS
from model.embeddings import RoPE2D


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with 2D RoPE attention."""

    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_drop_p = dropout

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope, h, w):
        """
        Args:
            x: [B, seq_len, embed_dim]
            rope: RoPE2D module
            h, w: grid dimensions
        Returns:
            [B, seq_len, embed_dim]
        """
        B, S, D = x.shape
        nh, hd = self.num_heads, self.head_dim

        # Pre-norm + QKV projection
        normed = self.norm1(x)
        qkv = self.qkv_proj(normed).reshape(B, S, 3, nh, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, nh, S, hd]
        q, k, v = qkv.unbind(0)

        # Apply 2D RoPE to Q and K
        q = rope(q, h, w)
        k = rope(k, h, w)

        # Scaled dot-product attention (uses Flash Attention when available)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )  # [B, nh, S, hd]

        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)
        x = x + self.dropout1(attn_out)

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer encoder blocks with 2D RoPE."""

    def __init__(self, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.rope = RoPE2D(embed_dim // num_heads)
        self.gradient_checkpointing = True

    def forward(self, x, h, w):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope, h, w, use_reentrant=False)
            else:
                x = layer(x, self.rope, h, w)
        return x
