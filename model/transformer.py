import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from config import (EMBED_DIM, NUM_HEADS, FFN_DIM, DROPOUT, NUM_LAYERS,
                    MAX_ROWS, MAX_COLS, TEXT_EMB_DIM)
from model.embeddings import RoPE2D


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer block: 2D-RoPE self-attention, cross-attention
    over caption tokens, feed-forward.

    The cross-attention output projection is zero-initialized, so the
    block behaves exactly like a plain encoder block until the caption
    pathway is trained — which also lets checkpoints from before
    cross-attention load with strict=False and run unchanged.
    """

    def __init__(self, embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT, ctx_dim=TEXT_EMB_DIM):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_drop_p = dropout

        # Self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention: grid queries attend to caption tokens
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.cross_q_proj = nn.Linear(embed_dim, embed_dim)
        self.cross_kv_proj = nn.Linear(ctx_dim, 2 * embed_dim)
        self.cross_out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_cross = nn.Dropout(dropout)
        nn.init.zeros_(self.cross_out_proj.weight)
        nn.init.zeros_(self.cross_out_proj.bias)

        # Feed-forward
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, rope, h, w, ctx, ctx_mask):
        """
        Args:
            x: [B, seq_len, embed_dim]
            rope: RoPE2D module
            h, w: grid dimensions
            ctx: [B, L, ctx_dim] caption-token context (the null token for
                 unconditional rows)
            ctx_mask: [B, L] bool, True = real token (False = padding)
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

        # Cross-attention over caption tokens (no RoPE: caption order is
        # encoded by the text encoder, grid cells query it content-wise)
        L = ctx.shape[1]
        cq = (self.cross_q_proj(self.norm_cross(x))
              .reshape(B, S, nh, hd).transpose(1, 2))       # [B, nh, S, hd]
        ckv = (self.cross_kv_proj(ctx)
               .reshape(B, L, 2, nh, hd).permute(2, 0, 3, 1, 4))
        ck, cv = ckv.unbind(0)                              # [B, nh, L, hd]
        cross_out = F.scaled_dot_product_attention(
            cq, ck, cv,
            attn_mask=ctx_mask[:, None, None, :],  # True = attend
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        cross_out = cross_out.transpose(1, 2).reshape(B, S, D)
        x = x + self.dropout_cross(self.cross_out_proj(cross_out))

        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of pre-norm transformer encoder blocks with 2D RoPE
    self-attention and caption-token cross-attention."""

    def __init__(self, num_layers=NUM_LAYERS, embed_dim=EMBED_DIM,
                 num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT,
                 ctx_dim=TEXT_EMB_DIM):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_dim, dropout,
                                    ctx_dim)
            for _ in range(num_layers)
        ])
        self.rope = RoPE2D(embed_dim // num_heads)
        self.gradient_checkpointing = True

    def forward(self, x, h, w, ctx, ctx_mask):
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, self.rope, h, w, ctx, ctx_mask,
                               use_reentrant=False)
            else:
                x = layer(x, self.rope, h, w, ctx, ctx_mask)
        return x
