import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, GRID_H, GRID_W,
    NUM_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT, NUM_CLASSES,
)
from model.embeddings import CombinedEmbedding, ConditioningEmbedding
from model.transformer import TransformerEncoder


class ASCIIBert(nn.Module):
    """
    MaskGIT-style model for ASCII art with 2D RoPE attention.

    Input:  [B, H, W] integer grid (some positions set to MASK_TOKEN)
            + optional class label and mask ratio (global conditioning)
    Output: [B, H, W, VOCAB_SIZE] logits for every grid position
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT, num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = CombinedEmbedding(vocab_size, embed_dim, dropout)
        self.conditioning = ConditioningEmbedding(num_classes, embed_dim)
        self.transformer = TransformerEncoder(num_layers, embed_dim,
                                              num_heads, ffn_dim, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, class_label=None, mask_ratio=None):
        """
        Args:
            x: [B, H, W] long tensor of token indices
            class_label: None | int | [B] long — global class conditioning
            mask_ratio: None | float | [B] float — fraction of grid masked
        Returns:
            logits: [B, H, W, VOCAB_SIZE]
        """
        B, H, W = x.shape
        cond = self.conditioning(B, x.device, class_label, mask_ratio)
        emb = self.embedding(x, cond)     # [B, H*W, D]
        out = self.transformer(emb, H, W) # [B, H*W, D]
        out = self.norm(out)              # [B, H*W, D]
        logits = self.head(out)           # [B, H*W, VOCAB_SIZE]
        logits = logits.view(B, H, W, -1) # [B, H, W, VOCAB_SIZE]
        return logits

    def compute_loss(self, logits, targets, mask, soft_target_matrix=None):
        """
        Cross-entropy loss computed only on masked positions.

        Args:
            logits:  [B, H, W, VOCAB_SIZE]
            targets: [B, H, W] long tensor (original unmasked grid)
            mask:    [B, H, W] bool tensor (True = masked position)
            soft_target_matrix: optional [VOCAB_SIZE, VOCAB_SIZE] tensor whose
                row t is the soft target distribution for true token t
                (glyph-aware label smoothing). None -> plain cross-entropy.
        Returns:
            scalar loss
        """
        masked_logits = logits[mask]    # [num_masked, VOCAB_SIZE]
        masked_targets = targets[mask]  # [num_masked]
        if masked_targets.numel() == 0:
            return logits.sum() * 0.0   # keep graph, no masked cells

        if soft_target_matrix is None:
            return F.cross_entropy(masked_logits, masked_targets)

        # Soft targets: gather each masked position's target distribution
        soft = soft_target_matrix.to(masked_logits.dtype)[masked_targets]
        log_probs = F.log_softmax(masked_logits, dim=-1)
        return -(soft * log_probs).sum(dim=-1).mean()
