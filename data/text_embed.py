"""Frozen text encoder for prompt conditioning.

Prompts (dataset captions, class names, user text) are embedded once with
a frozen CLIP text encoder; the resulting [TEXT_EMB_DIM] vectors are what
ASCIIBert conditions on. The encoder is never trained and is only needed
where new text appears — data preparation, the start of a conditioned
training stage (to embed the caption vocabulary), and interactive
generation — never inside the training loop itself.
"""

import torch

from config import TEXT_ENCODER, TEXT_EMB_DIM

_ENCODER = None


def _load_encoder(device="cpu"):
    global _ENCODER
    if _ENCODER is None:
        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as e:
            raise ImportError(
                "Text conditioning needs the 'transformers' package: "
                "pip install transformers") from e
        tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER)
        encoder = (CLIPTextModelWithProjection.from_pretrained(TEXT_ENCODER)
                   .to(device).eval())
        _ENCODER = (tokenizer, encoder, device)
    return _ENCODER


@torch.no_grad()
def embed_texts(texts, device="cpu", batch_size=256):
    """Embed strings -> [N, TEXT_EMB_DIM] float32 with unit-norm rows.

    Unit norm keeps the conditioning input scale constant so the learned
    projection doesn't have to absorb encoder-dependent magnitudes.
    """
    tokenizer, encoder, enc_device = _load_encoder(device)
    out = []
    for i in range(0, len(texts), batch_size):
        batch = tokenizer(list(texts[i:i + batch_size]), padding=True,
                          truncation=True, return_tensors="pt").to(enc_device)
        emb = encoder(**batch).text_embeds  # [B, TEXT_EMB_DIM]
        out.append(emb.float().cpu())
    emb = torch.cat(out)
    assert emb.shape[1] == TEXT_EMB_DIM, (
        f"{TEXT_ENCODER} produces {emb.shape[1]}-dim embeddings but config "
        f"TEXT_EMB_DIM={TEXT_EMB_DIM}")
    return emb / emb.norm(dim=1, keepdim=True).clamp_min(1e-8)
