import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, GRID_H, GRID_W,
    NUM_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT,
)
from model.embeddings import CombinedEmbedding
from model.transformer import TransformerEncoder


class ASCIIBert(nn.Module):
    """
    MaskGIT-style model for ASCII art with 2D RoPE attention.

    Input:  [B, H, W] integer grid (some positions set to MASK_TOKEN)
    Output: [B, H, W, VOCAB_SIZE] logits for every grid position
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT):
        super().__init__()
        self.embedding = CombinedEmbedding(vocab_size, embed_dim, dropout)
        self.transformer = TransformerEncoder(num_layers, embed_dim,
                                              num_heads, ffn_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        """
        Args:
            x: [B, H, W] long tensor of token indices
        Returns:
            logits: [B, H, W, VOCAB_SIZE]
        """
        B, H, W = x.shape
        emb = self.embedding(x)           # [B, H*W, D]
        out = self.transformer(emb, H, W) # [B, H*W, D]
        out = self.norm(out)              # [B, H*W, D]
        logits = self.head(out)           # [B, H*W, VOCAB_SIZE]
        logits = logits.view(B, H, W, -1) # [B, H, W, VOCAB_SIZE]
        return logits

    def compute_loss(self, logits, targets, mask):
        """
        Cross-entropy loss computed only on masked positions.

        Args:
            logits:  [B, H, W, VOCAB_SIZE]
            targets: [B, H, W] long tensor (original unmasked grid)
            mask:    [B, H, W] bool tensor (True = masked position)
        Returns:
            scalar loss
        """
        # Gather only masked positions
        masked_logits = logits[mask]    # [num_masked, VOCAB_SIZE]
        masked_targets = targets[mask]  # [num_masked]
        return F.cross_entropy(masked_logits, masked_targets)
