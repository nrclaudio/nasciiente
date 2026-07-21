"""FastAPI backend for the showcase site (app/static/index.html).

Serves the model over a small JSON API and the static frontend at /.
The frontend animates the MaskGIT unmasking steps client-side, so the
API returns every intermediate grid, not just the final one.

Run:
    uvicorn app.server:app --host 0.0.0.0 --port 8501
    # or: python -m app.server

Env:
    ASCII_CHECKPOINT  path override (default: checkpoints/final_model.pt,
                      falling back to the latest .pt in checkpoints/)
    ASCII_DEVICE      cpu|cuda override (default: cuda when >3GB free)
"""

import os
import sys
import time

# Must be set before torch loads: lets any op MPS lacks fall back to
# CPU instead of crashing the request (harmless elsewhere)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE, CFG_SCALE,
                    MAX_ROWS, MAX_COLS, CFG_SCHEDULE, DECODE_MAX_COMMIT,
                    RANKER_MODEL)
from data.charset import grid_to_string
from model.ascii_bert import (ASCIIBert, load_compatible_state,
                              model_matching_state)
from model.inference import generate

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import Response
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise ImportError("The showcase site needs: pip install fastapi "
                      "uvicorn python-multipart") from e

_STATE = {}


def _find_checkpoint():
    env = os.environ.get("ASCII_CHECKPOINT")
    if env:
        return env
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    final = os.path.join(ckpt_dir, "final_model.pt")
    if os.path.exists(final):
        return final
    if os.path.isdir(ckpt_dir):
        pts = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
        if pts:
            return os.path.join(ckpt_dir, pts[-1])
    return None


def _pick_device():
    env = os.environ.get("ASCII_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > 3e9:
                return torch.device("cuda")
        except Exception:
            pass
    # Apple GPU: several times faster than CPU for this transformer.
    # ASCII_DEVICE=cpu overrides if an MPS op misbehaves.
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model():
    if "model" not in _STATE:
        path = _find_checkpoint()
        if path is None:
            raise HTTPException(503, "No checkpoint found in checkpoints/")
        device = _pick_device()
        ckpt = torch.load(path, map_location=device, weights_only=True)
        state = (ckpt["model_state_dict"]
                 if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                 else ckpt)
        model = model_matching_state(state).to(device)
        conditioned = load_compatible_state(model, state)
        model.eval()
        # Opt-in int8 dynamic quantization for CPU serving (~1.5-2x on
        # the Linear-dominated forward). Off by default: the decode
        # ranks commits by confidence, and quantized logits haven't
        # been probe-validated against the fp32 outputs.
        if (device.type == "cpu"
                and os.environ.get("ASCII_QUANTIZE") == "1"):
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        _STATE.update(model=model, device=device, conditioned=conditioned,
                      checkpoint=os.path.basename(path))
    return _STATE


app = FastAPI(title="nASCIIente")


class GenRequest(BaseModel):
    prompt: str = ""
    guidance: float = Field(CFG_SCALE, ge=1.0, le=8.0)
    # 32 is the probe-validated step count; the old cap of 20 couldn't
    # even request it
    steps: int = Field(32, ge=1, le=64)
    temperature: float = Field(TEMPERATURE, ge=0.1, le=2.0)
    rows: int = Field(GRID_H, ge=8, le=MAX_ROWS)
    cols: int = Field(GRID_W, ge=8, le=MAX_COLS)
    seed: int | None = None
    variations: int = Field(1, ge=1, le=4)
    space_bias: float = Field(0.0, ge=0.0, le=8.0)
    # 0: the grid-validated decode uses no revision passes (they re-roll
    # parallel commits rather than repair — findings table)
    revision_steps: int = Field(0, ge=0, le=5)
    schedule: str = Field(CFG_SCHEDULE, pattern="^(constant|rise|fall)$")
    # commit-order exploration noise: 1 = organic/tonal, 0 = precise
    gumbel: float = Field(1.0, ge=0.0, le=2.0)
    # cells committed per decode step; None = uncapped (fast, echo-prone)
    max_commit: int | None = Field(DECODE_MAX_COMMIT, ge=1, le=512)


@app.get("/api/info")
def info():
    s = get_model()
    n_params = sum(p.numel() for p in s["model"].parameters())
    return {"checkpoint": s["checkpoint"],
            "params_m": round(n_params / 1e6, 1),
            "conditioned": s["conditioned"],
            "device": s["device"].type,
            "grid": [GRID_H, GRID_W]}


@app.post("/api/generate")
def api_generate(req: GenRequest):
    s = get_model()
    model, device, conditioned = s["model"], s["device"], s["conditioned"]

    kwargs = dict(space_bias=req.space_bias,
                  revision_steps=req.revision_steps,
                  guidance_schedule=req.schedule,
                  max_commit=req.max_commit,
                  gumbel_scale=req.gumbel)
    prompt = req.prompt.strip()
    if prompt and conditioned:
        from data.text_embed import embed_captions
        toks, msk = embed_captions([prompt])
        kwargs.update(cond_tokens=toks[0], cond_mask=msk[0],
                      guidance_scale=req.guidance)

    results = []
    for i in range(req.variations):
        seed = (req.seed + i if req.seed is not None
                else int(time.time_ns() % 2**31) ^ (i * 7919))
        torch.manual_seed(seed)
        t0 = time.time()
        steps, final = generate(model, req.rows, req.cols,
                                num_steps=req.steps,
                                temperature=req.temperature,
                                device=device, **kwargs)
        results.append({"seed": seed,
                        "took": round(time.time() - t0, 2),
                        "steps": [grid_to_string(g) for g in steps],
                        "final": grid_to_string(final),
                        "grids": None,
                        "score": None,
                        "_grid": final})

    # Re-rank best-of-k only when an ASCII-aware ranker is configured:
    # vanilla CLIP scores rendered ASCII at chance level (ASCIIBench),
    # so without one the sort is noise that costs a full CLIP vision
    # forward per run — pure latency on CPU serving.
    ranked = False
    if prompt and conditioned and len(results) > 1 and RANKER_MODEL:
        try:
            from data.clip_rank import clip_scores
            scores = clip_scores([r["_grid"] for r in results], prompt)
            for r, sc in zip(results, scores):
                r["score"] = round(float(sc), 4)
            results.sort(key=lambda r: r["score"], reverse=True)
            ranked = True
        except Exception:
            pass
    for r in results:
        r.pop("_grid")
    return {"results": results,
            "conditioned": conditioned,
            "ranked": ranked,
            "prompt_used": bool(prompt and conditioned)}


MAX_CLOUD_FRAMES = 48


@app.post("/api/cloud")
def api_cloud(req: GenRequest):
    """Probability-cloud replay: re-run one decode (same seed = same
    trajectory as a previous /api/generate result) capturing the full
    per-cell glyph distribution at every step, and render each step as
    the EXPECTATION over glyphs — committed cells sharp, undecided
    cells as their probability-weighted ghost. The returned GIF is the
    superposition collapsing into the final art."""
    import io as _io

    from PIL import Image

    from data.charset import MASK_TOKEN
    from model.render import render_grid, render_probs

    s = get_model()
    model, device, conditioned = s["model"], s["device"], s["conditioned"]

    kwargs = dict(space_bias=req.space_bias,
                  revision_steps=req.revision_steps,
                  guidance_schedule=req.schedule,
                  max_commit=req.max_commit,
                  gumbel_scale=req.gumbel)
    prompt = req.prompt.strip()
    if prompt and conditioned:
        from data.text_embed import embed_captions
        toks, msk = embed_captions([prompt])
        kwargs.update(cond_tokens=toks[0], cond_mask=msk[0],
                      guidance_scale=req.guidance)
    if not prompt and req.space_bias == 0:
        kwargs["space_bias"] = 3.0  # match the frontend's unprompted default

    seed = (req.seed if req.seed is not None
            else int(time.time_ns() % 2**31))
    torch.manual_seed(seed)
    captured = []
    _, final = generate(model, req.rows, req.cols, num_steps=req.steps,
                        temperature=req.temperature, device=device,
                        probs_out=captured, **kwargs)

    # A capped tail can run hundreds of steps — subsample evenly
    if len(captured) > MAX_CLOUD_FRAMES:
        idx = [round(i * (len(captured) - 1) / (MAX_CLOUD_FRAMES - 1))
               for i in range(MAX_CLOUD_FRAMES)]
        captured = [captured[i] for i in idx]

    frames = []
    for pre_grid, probs in captured:
        onehot = torch.nn.functional.one_hot(
            pre_grid, num_classes=probs.shape[-1]).float()
        composite = torch.where((pre_grid == MASK_TOKEN).unsqueeze(-1),
                                probs.float(), onehot)
        img = render_probs(composite).clamp(0, 1)
        # gamma-lift the faint expectation ink so the early cloud reads
        img = img ** 0.5
        arr = ((1.0 - img) * 255).byte().numpy()
        frames.append(Image.fromarray(arr).convert("P"))
    # hold the sharp final render
    arr = ((1.0 - render_grid(final).clamp(0, 1)) * 255).byte().numpy()
    frames.extend([Image.fromarray(arr).convert("P")] * 8)

    buf = _io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=150, loop=0)
    return Response(buf.getvalue(), media_type="image/gif",
                    headers={"X-Seed": str(seed)})


def _converter_tables():
    if "tables" not in _STATE:
        import contextlib
        import io as _io
        from data.generate_synthetic import _build_tables
        with contextlib.redirect_stdout(_io.StringIO()):
            _STATE["tables"] = _build_tables()
    return _STATE["tables"]


MAX_FRAMES = 60


@app.post("/api/convert")
async def api_convert(file: UploadFile = File(...), binarize: bool = True):
    """Convert an uploaded image (or animated GIF, frame by frame) to
    ASCII via the 6D shape-matching converter used for training data."""
    import io as _io

    from PIL import Image, ImageSequence

    from data.generate_synthetic import images_to_grids

    raw = await file.read()
    if len(raw) > 20e6:
        raise HTTPException(413, "File too large (20MB max)")
    try:
        img = Image.open(_io.BytesIO(raw))
    except Exception:
        raise HTTPException(422, "Not a readable image")

    frames = [f.convert("RGB") for f in
              ImageSequence.Iterator(img)][:MAX_FRAMES]
    converted = images_to_grids(frames, _converter_tables(),
                                min_ink=0.0, max_ink=1.0,
                                binarize=binarize)
    texts = [grid_to_string(g.long()) for g, _ in converted]
    fps = 10
    if getattr(img, "is_animated", False):
        duration = img.info.get("duration") or 100
        fps = max(1, min(20, round(1000 / max(duration, 1))))
    return {"frames": texts, "fps": fps,
            "animated": len(texts) > 1}


class GifRequest(BaseModel):
    frames: list[str] = Field(..., min_length=1, max_length=120)
    fps: float = Field(8, ge=1, le=30)


@app.post("/api/gif")
def api_gif(req: GifRequest):
    """Render ASCII frames to an animated GIF via the glyph atlas —
    makes any animation on the site (materialize replays, converted
    GIFs) exportable and shareable."""
    import io as _io

    from PIL import Image

    from data.charset import string_to_grid
    from model.render import render_grid

    images = []
    for text in req.frames:
        grid = string_to_grid(text)
        arr = ((1.0 - render_grid(grid).clamp(0, 1)) * 255).byte().numpy()
        images.append(Image.fromarray(arr).convert("P"))
    buf = _io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True,
                   append_images=images[1:],
                   duration=int(1000 / req.fps), loop=0)
    return Response(buf.getvalue(), media_type="image/gif")


@app.get("/api/progress")
def progress():
    sample_dir = os.path.join(os.path.dirname(__file__), "..",
                              "checkpoints", "samples")
    import glob as _glob
    entries = []
    for path in sorted(_glob.glob(os.path.join(sample_dir,
                                               "*_epoch*.txt"))):
        stage, _, epoch = os.path.basename(path)[:-4].rpartition("_epoch")
        try:
            with open(path) as f:
                entries.append({"stage": stage, "epoch": int(epoch),
                                "text": f.read()})
        except (ValueError, OSError):
            continue
    return {"samples": entries}


app.mount("/", StaticFiles(
    directory=os.path.join(os.path.dirname(__file__), "static"),
    html=True), name="static")


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
