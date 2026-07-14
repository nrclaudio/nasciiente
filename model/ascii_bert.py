import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    VOCAB_SIZE, EMBED_DIM, GRID_H, GRID_W,
    NUM_LAYERS, NUM_HEADS, FFN_DIM, DROPOUT, TEXT_EMB_DIM,
    SPACE_LOSS_WEIGHT, PERCEPTUAL_LOSS_WEIGHT,
)
from data.charset import char_to_idx

SPACE_IDX = char_to_idx(" ")
from model.embeddings import CombinedEmbedding, ConditioningEmbedding
from model.transformer import TransformerEncoder

# Parameters belonging to the (zero-init, no-op-until-trained) conditioning
# pathway: the ConditioningEmbedding module and the cross-attention
# sublayers. Checkpoints from before text conditioning lack these keys and
# may carry obsolete ones — both are safe to tolerate when loading.
_COND_KEY_MARKERS = ("conditioning.", "cross_", "norm_cross")

# Later zero-init retrofits (same pattern as the conditioning splice):
# tolerated when loading older checkpoints, but NOT counted as evidence
# about text conditioning. The critic head predicts, per cell, whether
# the visible glyph is correct (Token-Critic, arXiv 2209.04439) — used
# at decode to rank commits/revisions by a TRAINED signal instead of
# generator confidence.
_RETROFIT_KEY_MARKERS = ("critic.",)


def _is_conditioning_key(key):
    return any(m in key for m in _COND_KEY_MARKERS)


def _is_retrofit_key(key):
    return any(m in key for m in _RETROFIT_KEY_MARKERS)


# Named model sizes. "base" is the original 34.5M configuration;
# "large" (~113M) exists for the tonal dialect, whose per-cell entropy
# is far higher than geometry/silhouette work.
MODEL_SIZES = {
    "base":  dict(embed_dim=512, num_layers=8, num_heads=8, ffn_dim=2048),
    "large": dict(embed_dim=768, num_layers=12, num_heads=12,
                  ffn_dim=3072),
}


def arch_from_state(state):
    """Infer constructor kwargs from a checkpoint's tensor shapes, so
    tools (probe, server) can load any model size without being told.
    num_heads follows the head_dim=64 convention both sizes use."""
    embed_dim = state["embedding.token_embed.embedding.weight"].shape[1]
    return dict(
        embed_dim=embed_dim,
        num_layers=1 + max(int(k.split(".")[2]) for k in state
                           if k.startswith("transformer.layers.")),
        num_heads=max(1, embed_dim // 64),
        ffn_dim=state["transformer.layers.0.ffn.0.weight"].shape[0],
        text_dim=state["conditioning.null_token"].shape[0],
    )


def model_matching_state(state):
    """Construct an ASCIIBert shaped like the checkpoint."""
    return ASCIIBert(**arch_from_state(state))


def load_compatible_state(model, state):
    """Load a checkpoint state dict, tolerating only conditioning-pathway
    mismatches (older checkpoints predate text conditioning; those modules
    are zero-init no-ops so the model behaves as it did then). Any other
    mismatch raises instead of silently leaving random weights.

    Returns True if the checkpoint carries trained conditioning params.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in list(missing) + list(unexpected)
           if not (_is_conditioning_key(k) or _is_retrofit_key(k))]
    if bad:
        raise ValueError(f"Checkpoint does not match the model "
                         f"(mismatched keys: {bad[:5]}...)")
    return not any(_is_conditioning_key(k) for k in missing)


class ASCIIBert(nn.Module):
    """
    MaskGIT-style model for ASCII art with 2D RoPE attention.

    Input:  [B, H, W] integer grid (some positions set to MASK_TOKEN)
            + optional text-prompt embedding and mask ratio (global
            conditioning)
    Output: [B, H, W, VOCAB_SIZE] logits for every grid position
    """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS,
                 ffn_dim=FFN_DIM, dropout=DROPOUT, text_dim=TEXT_EMB_DIM):
        super().__init__()
        self.embedding = CombinedEmbedding(vocab_size, embed_dim, dropout)
        self.conditioning = ConditioningEmbedding(text_dim, embed_dim)
        self.transformer = TransformerEncoder(num_layers, embed_dim,
                                              num_heads, ffn_dim, dropout,
                                              ctx_dim=text_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        # Token-Critic head: per-cell logit for "the glyph visible at this
        # cell is correct". Zero-init -> outputs 0 (sigmoid 0.5, neutral)
        # until trained, so it can be retrofitted onto any checkpoint.
        # Trained from the self-context corruption's revealed cells, whose
        # correct/incorrect labels the training loop gets for free.
        self.critic = nn.Linear(embed_dim, 1)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x, cond_tokens=None, cond_mask=None, cond_drop=None,
                mask_ratio=None, return_critic=False):
        """
        Args:
            x: [B, H, W] long tensor of token indices
            cond_tokens: None | [L, text_dim] | [B, L, text_dim] float —
                         caption-token embeddings from the frozen text
                         encoder (None -> the learned null token)
            cond_mask: optional [L] or [B, L] bool, True = real token
            cond_drop: optional [B] bool — rows to force to the null token
                       (CFG dropout / unconditional branch)
            mask_ratio: None | float | [B] float — fraction of grid masked
            return_critic: also return the per-cell critic logits
        Returns:
            logits: [B, H, W, VOCAB_SIZE]
            (logits, critic): if return_critic — critic is [B, H, W]
        """
        B, H, W = x.shape
        cond_vec, ctx, ctx_mask = self.conditioning(
            B, x.device, cond_tokens, cond_mask, cond_drop, mask_ratio)
        emb = self.embedding(x, cond_vec)              # [B, H*W, D]
        out = self.transformer(emb, H, W, ctx, ctx_mask)  # [B, H*W, D]
        out = self.norm(out)              # [B, H*W, D]
        logits = self.head(out)           # [B, H*W, VOCAB_SIZE]
        logits = logits.view(B, H, W, -1) # [B, H, W, VOCAB_SIZE]
        if return_critic:
            critic = self.critic(out).view(B, H, W)
            return logits, critic
        return logits

    def compute_loss(self, logits, targets, mask, soft_target_matrix=None,
                     space_weight=SPACE_LOSS_WEIGHT,
                     perceptual_weight=PERCEPTUAL_LOSS_WEIGHT):
        """
        Cross-entropy loss computed only on masked positions, plus an
        optional perceptual term through the differentiable glyph
        renderer.

        Args:
            logits:  [B, H, W, VOCAB_SIZE]
            targets: [B, H, W] long tensor (original unmasked grid)
            mask:    [B, H, W] bool tensor (True = masked position)
            soft_target_matrix: optional [VOCAB_SIZE, VOCAB_SIZE] tensor whose
                row t is the soft target distribution for true token t
                (glyph-aware label smoothing). None -> plain cross-entropy.
            space_weight: weight applied to cells whose TRUE token is space
                (<1 counteracts the space-majority prior that drives
                blank-collapse in free generation; 1.0 = plain mean)
            perceptual_weight: weight of the rendered-pixel MSE between
                the expected glyph bitmap under the predicted distribution
                and the true glyph's bitmap. CE charges every wrong glyph
                full price; this term charges by visual distance — the
                criterion ASCII art is actually judged by. 0 disables.
        Returns:
            scalar loss
        """
        masked_logits = logits[mask]    # [num_masked, VOCAB_SIZE]
        masked_targets = targets[mask]  # [num_masked]
        if masked_targets.numel() == 0:
            return logits.sum() * 0.0   # keep graph, no masked cells

        if soft_target_matrix is None:
            per_cell = F.cross_entropy(masked_logits, masked_targets,
                                       reduction="none")
        else:
            # Soft targets: gather each position's target distribution
            soft = soft_target_matrix.to(masked_logits.dtype)[masked_targets]
            log_probs = F.log_softmax(masked_logits, dim=-1)
            per_cell = -(soft * log_probs).sum(dim=-1)

        if space_weight == 1.0:
            loss = per_cell.mean()
        else:
            weights = torch.where(masked_targets == SPACE_IDX,
                                  torch.full_like(per_cell, space_weight),
                                  torch.ones_like(per_cell))
            # Weighted mean keeps the loss scale comparable across batches
            loss = ((per_cell * weights).sum()
                    / weights.sum().clamp_min(1e-8))

        if perceptual_weight > 0:
            atlas = self._glyph_atlas(masked_logits.device,
                                      masked_logits.dtype)
            if atlas is not None:
                probs = masked_logits.softmax(-1)          # [N, V]
                v = atlas.shape[0]
                flat = atlas.reshape(v, -1)                # [V, ch*cw]
                pred_px = probs @ flat                     # [N, ch*cw]
                target_px = flat[masked_targets]           # [N, ch*cw]
                loss = loss + perceptual_weight * F.mse_loss(pred_px,
                                                             target_px)
        return loss

    def _glyph_atlas(self, device, dtype):
        """Cached glyph atlas for the perceptual loss (None if no font
        is available — the term silently disables rather than training
        against blank bitmaps)."""
        cached = getattr(self, "_atlas_cache", None)
        if cached is None:
            from model.render import glyph_atlas
            cached = glyph_atlas()
            if cached.sum() == 0:
                cached = False        # sentinel: no font on this machine
            self._atlas_cache = cached
        if cached is False:
            return None
        return cached.to(device=device, dtype=dtype)
