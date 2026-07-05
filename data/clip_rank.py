"""CLIP re-ranking: score rendered grids against a prompt.

Generate several candidates, render each to pixels, and ask CLIP which
render actually looks like the prompt — the first rung of judging ASCII
art by its appearance instead of its token likelihood. Uses the full CLIP
model (vision tower included), loaded lazily and cached; the text-only
encoder in text_embed.py stays lightweight for training.
"""

import torch

from config import TEXT_ENCODER

_SCORER = None


def _load_scorer(device="cpu"):
    global _SCORER
    if _SCORER is None:
        try:
            from transformers import AutoProcessor, CLIPModel
        except ImportError as e:
            raise ImportError(
                "CLIP ranking needs the 'transformers' package: "
                "pip install transformers") from e
        model = CLIPModel.from_pretrained(TEXT_ENCODER).to(device).eval()
        processor = AutoProcessor.from_pretrained(TEXT_ENCODER)
        _SCORER = (model, processor, device)
    return _SCORER


@torch.no_grad()
def clip_scores(grids, prompt, device="cpu"):
    """Cosine similarity of each rendered grid to the prompt.

    Args:
        grids: list of [H, W] long tensors
        prompt: the text the art is supposed to depict
    Returns:
        [len(grids)] float tensor (higher = renders more like the prompt)
    """
    from model.render import render_to_pil
    model, processor, dev = _load_scorer(device)
    images = [render_to_pil(g) for g in grids]
    inputs = processor(text=[prompt], images=images, return_tensors="pt",
                       padding=True, truncation=True).to(dev)
    out = model(**inputs)
    # image_embeds/text_embeds are already L2-normalized
    return (out.image_embeds @ out.text_embeds.T).squeeze(-1).float().cpu()
