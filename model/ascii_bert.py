import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VOCAB_SIZE, EMBED_DIM, GRID_H, GRID_W
from model.embeddings import CombinedEmbedding
from model.transformer import TransformerEncoder


class ASCIIBert(nn.Module):
    """
    MaskGIT-style model for ASCII art with 2D RoPE attention.

    Input:  [B, H, W] integer grid (some positions set to MASK_TOKEN)
    Output: [B, H, W, VOCAB_SIZE] logits for every grid position
    """

    def __init__(self):
        super().__init__()
        self.embedding = CombinedEmbedding()
        self.transformer = TransformerEncoder()
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

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
