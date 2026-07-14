"""CLIP re-ranking: score rendered grids against a prompt.

Generate several candidates, render each to pixels, and ask CLIP which
render actually looks like the prompt — the first rung of judging ASCII
art by its appearance instead of its token likelihood. Uses the full CLIP
model (vision tower included), loaded lazily and cached; the text-only
encoder in text_embed.py stays lightweight for training.
"""

import torch

from config import TEXT_ENCODER, RANKER_MODEL

_SCORERS = {}


def _load_scorer(device="cpu", model_id=None):
    """Load (and cache) a CLIP scorer. model_id=None -> TEXT_ENCODER."""
    model_id = model_id or TEXT_ENCODER
    key = (model_id, device)
    if key not in _SCORERS:
        try:
            from transformers import AutoProcessor, CLIPModel
        except ImportError as e:
            raise ImportError(
                "CLIP ranking needs the 'transformers' package: "
                "pip install transformers") from e
        model = CLIPModel.from_pretrained(model_id).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_id)
        _SCORERS[key] = (model, processor, device)
    return _SCORERS[key]


@torch.no_grad()
def clip_image_scores(pil_images, captions, device="cpu"):
    """Cosine similarity between each PIL image and ITS OWN caption.

    The data engine's consistency filter: generations that don't depict
    their prompt (subject misses, degenerate outputs) score low and get
    dropped before they become wrong training labels.
    """
    model, processor, dev = _load_scorer(device)
    inputs = processor(text=list(captions), images=list(pil_images),
                       return_tensors="pt", padding=True,
                       truncation=True).to(dev)
    out = model(**inputs)
    return (out.image_embeds * out.text_embeds).sum(-1).float().cpu()


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
    # Rendered-ASCII scoring uses RANKER_MODEL when set: vanilla CLIP is
    # near chance level on ASCII structure (arXiv 2503.08295), so an
    # ASCII-aware checkpoint materially changes which candidate wins.
    model, processor, dev = _load_scorer(device, RANKER_MODEL)
    images = [render_to_pil(g) for g in grids]
    inputs = processor(text=[prompt], images=images, return_tensors="pt",
                       padding=True, truncation=True).to(dev)
    out = model(**inputs)
    # image_embeds/text_embeds are already L2-normalized
    return (out.image_embeds @ out.text_embeds.T).squeeze(-1).float().cpu()
