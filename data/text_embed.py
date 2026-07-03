"""Frozen text encoder for prompt conditioning.

Prompts (dataset captions, class names, user text) are embedded once with
a frozen CLIP text encoder; the per-token hidden states are what ASCIIBert
cross-attends to (with their masked mean feeding the global conditioning
vector). The encoder is never trained and is only needed where new text
appears — data preparation, the start of a conditioned training stage (to
embed the caption vocabulary), and interactive generation — never inside
the training loop itself.
"""

import torch

from config import TEXT_ENCODER, TEXT_EMB_DIM, TEXT_COND_TOKENS

_ENCODER = None


def _load_encoder(device="cpu"):
    global _ENCODER
    if _ENCODER is None:
        try:
            from transformers import AutoTokenizer, CLIPTextModel
        except ImportError as e:
            raise ImportError(
                "Text conditioning needs the 'transformers' package: "
                "pip install transformers") from e
        tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
        encoder = (CLIPTextModel.from_pretrained(TEXT_ENCODER)
                   .to(device).eval())
        _ENCODER = (tokenizer, encoder, device)
    return _ENCODER


@torch.no_grad()
def embed_captions(texts, device="cpu", batch_size=256):
    """Embed strings -> (tokens [N, L, TEXT_EMB_DIM] float32, mask [N, L]).

    tokens are the encoder's final per-token hidden states, RMS-normalized
    per token so the conditioning input scale is constant; mask is True on
    real tokens (incl. BOS/EOS), False on padding. L is the longest caption
    in `texts`, capped at TEXT_COND_TOKENS.
    """
    tokenizer, encoder, enc_device = _load_encoder(device)
    all_tokens, all_masks = [], []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[i:i + batch_size]), padding=True,
                          truncation=True, max_length=TEXT_COND_TOKENS,
                          return_tensors="pt").to(enc_device)
        hidden = encoder(**batch).last_hidden_state  # [B, L, D]
        all_tokens.append(hidden.float().cpu())
        all_masks.append(batch["attention_mask"].bool().cpu())

    # Pad every chunk to the global max length so they stack
    L = max(t.shape[1] for t in all_tokens)
    tokens = torch.zeros(len(texts), L, all_tokens[0].shape[2])
    mask = torch.zeros(len(texts), L, dtype=torch.bool)
    row = 0
    for t, m in zip(all_tokens, all_masks):
        tokens[row:row + t.shape[0], :t.shape[1]] = t
        mask[row:row + m.shape[0], :m.shape[1]] = m
        row += t.shape[0]

    assert tokens.shape[2] == TEXT_EMB_DIM, (
        f"{TEXT_ENCODER} produces {tokens.shape[2]}-dim hidden states but "
        f"config TEXT_EMB_DIM={TEXT_EMB_DIM}")
    # Per-token RMS normalization; padding positions zeroed (the encoder
    # emits non-zero states there, but they must never carry signal)
    rms = tokens.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    tokens = torch.where(mask.unsqueeze(-1), tokens / rms,
                         torch.zeros_like(tokens))
    return tokens, mask
