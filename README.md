# ascii-art-transformer

A BERT-style masked-token transformer that generates ASCII art on a 48×80
character grid, MaskGIT-style: start from a fully (or partially) masked grid
and iteratively unmask the most confident positions until the picture is
complete. Because the model is bidirectional, the same network does both
free generation and inpainting (fill in the blanks around user-provided
characters).

```
fully masked grid ──▶ ASCIIBert ──▶ logits per cell ──▶ unmask top-k most
        ▲                                                confident cells
        └────────────────── repeat ~10 steps ──────────────────┘
```

## Architecture

- **ASCIIBert** (`model/ascii_bert.py`) — ~25M parameters with the default
  config: 8 pre-norm transformer encoder layers, 512 dim, 8 heads, GELU FFN
  (2048), gradient checkpointing.
- **2D RoPE** (`model/embeddings.py`) — each attention head splits in half;
  the first half rotates by row position, the second by column position.
  No learned positional embeddings, so the model runs on any grid size up
  to 64×128 (`MAX_ROWS`/`MAX_COLS`).
- **Vocabulary** (`data/charset.py`) — 95 printable ASCII chars (indices
  2–96) plus `[PAD]`(0) and `[MASK]`(1).
- **Training objective** (`training/masking.py`) — mask a uniform-random
  15–85% of cells, cross-entropy on masked positions only. The wide mask
  ratio range is what makes MaskGIT-style inference work: early inference
  steps look like 85% masking, late steps like 15%.
- **Inference** (`model/inference.py`) — iterative unmasking with a linear
  schedule; per-step confidence = max softmax probability; sampling with
  temperature. Positions given in `initial_grid` are never overwritten
  (inpainting).

## Training curriculum

1. **Geometry** (`data/generate_geometry.py`) — 200k synthetic grids of
   lines, rectangles, triangles, crosses, diamonds and short text labels.
   Teaches structural primitives cheaply.
2. **Shading** (`data/generate_shading.py`) — 100k grids derived from
   images. Default source is **ImageNet-Sketch** (~50k line drawings,
   streamed from HuggingFace, no auth) — sketches convert to much cleaner
   ASCII than photos and match the geometry stage's distribution; photo
   sources (ImageNet / Imagenette / Caltech-101/256 / STL-10) remain
   available via `--source`. Each image is converted with 6D sub-cell
   shape matching: every cell is sampled by 6 circular regions and matched
   to the closest glyph in that 6D space, after adaptive gamma → CLAHE →
   Sobel edge blending. The converter letterboxes to preserve aspect
   ratio, auto-flips polarity so backgrounds render sparse, and floors
   low-ink pixels to true whitespace. For busy photos, `--segment`
   additionally removes backgrounds with rembg (optional dependency:
   `pip install rembg onnxruntime`). See `data/DATASET_OPTIONS.md` for the
   dataset comparison.
3. **Human ASCII art** (`data/prepare_human_ascii.py`, optional) —
   fine-tune on ~5k human-made pieces (HuggingFace `apehex/ascii-art`,
   scraped from asciiart.eu), normalized to 48×80 and filtered to the
   95-char vocabulary. Runs automatically as stage 3 if
   `data/human_data.pt` exists; skipped otherwise.

## Quickstart

```bash
pip install -r requirements.txt

# Run the test suite (CPU, a few seconds)
pytest tests/

# Full pipeline on a GPU box (generates data, then trains all stages)
python main.py                       # designed for Vast.ai instances
python main.py --stage geometry      # or run stages individually
python main.py --stage shading --source imagenette
python main.py --stage human         # optional stage-3 human ASCII art
python main.py --stage train

# SLURM clusters
sbatch train_slurm.sh

# Evaluate a checkpoint / inpainting demo
python training/evaluate.py --checkpoint checkpoints/final_model.pt \
                            --data data/geometry_data.pt

# Web UI (needs a trained checkpoint in checkpoints/)
streamlit run app/streamlit_app.py
```

Notes:
- The default shading source `imagenet_sketch` streams via HuggingFace
  with no auth. `--source imagenet` needs a HF token with ImageNet
  access; `imagenette`/`caltech101`/`caltech256` need no auth.
- Training defaults (`config.py`) target a single A100: batch 64,
  15 epochs per stage, cosine schedule with warmup. Checkpoints are
  written to `checkpoints/` every epoch, plus `final_model.pt`.
- Sample shading conversions are in `samples/focused/` (image / .txt pairs).

## Repository layout

```
config.py                 all hyperparameters
main.py                   single entry point (generate + train)
data/
  charset.py              vocab and grid<->string helpers
  generate_geometry.py    stage-1 synthetic data
  generate_shading.py     stage-2 image-derived data (6D shape matching)
  prepare_human_ascii.py  stage-3 human ASCII art (optional)
model/
  ascii_bert.py           the model
  embeddings.py           token embedding + 2D RoPE
  transformer.py          pre-norm encoder stack (SDPA/Flash attention)
  inference.py            MaskGIT-style iterative unmasking
training/
  masking.py, dataset.py  random masking + Dataset wrapper
  train.py                two-stage training loop
  evaluate.py             val metrics, sample generation, inpainting demo
app/
  streamlit_app.py        web UI (generate + inpainting)
tests/                    CPU-fast pytest suite
```

## Project status

- ✅ Full pipeline implemented and covered by tests (data generation,
  masking, model forward/backward, iterative inference, inpainting, app
  utils, shading conversion).
- ⬜ **Not yet done: the actual GPU training run.** No trained checkpoint
  is committed; the Streamlit app needs one. `main.py` (Vast.ai) or
  `train_slurm.sh` (SLURM) are ready to go — expect a few hours on one
  A100 for both stages.

## Roadmap

Next steps, roughly in order:

1. **Train the model** on a GPU (see Quickstart) and eyeball per-epoch
   samples; tune `SHADING_LR`/epochs if stage 2 forgets stage-1 structure,
   and `HUMAN_LR`/epochs if stage 3 overfits its small dataset.
2. **Publish a checkpoint** (GitHub release or HF Hub) so the Streamlit
   app works out of the box.
3. **Better unmasking schedule** — MaskGIT's cosine schedule and
   confidence-based *re*masking usually beat the current linear schedule.
4. **Vectorize inference sampling** — `model/inference.py` samples masked
   cells in a Python loop; a batched `torch.multinomial` would speed up
   the app noticeably on CPU.
5. **Conditioning** — class- or text-conditioned generation (prepend a
   learned condition token, or cross-attend to a caption embedding) would
   make generation controllable instead of unconditional.
6. **Larger grids** — RoPE already supports up to 64×128; train with
   variable grid sizes to exploit it.
7. **More human data** — `mrzjy/ascii_art_generation_140k` (139k
   image-derived pieces) and `Csplk/THE.ASCII.ART.EMPORIUM` (3.1M scraped
   rows) on HuggingFace are candidate stage-3 expansions after filtering.
