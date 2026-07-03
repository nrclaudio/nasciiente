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

- **ASCIIBert** (`model/ascii_bert.py`) — ~35M parameters with the default
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

# SLURM clusters (multi-GPU: raise --gres=gpu:N in the script)
sbatch train_slurm.sh

# Multi-GPU on any box (main.py does this automatically when it sees >1 GPU)
torchrun --standalone --nproc_per_node=4 training/train.py

# Evaluate a checkpoint / inpainting demo
python training/evaluate.py --checkpoint checkpoints/final_model.pt \
                            --data data/geometry_data.pt

# Web UI (needs a trained checkpoint in checkpoints/)
streamlit run app/streamlit_app.py
```

### Resource requirements (single full run)

| Resource | Minimum | Comfortable | Notes |
|----------|---------|-------------|-------|
| GPU VRAM | 16 GB | 24 GB (4090) / 40+ GB (A100) | gradient checkpointing is on; if OOM, halve `BATCH_SIZE` and set `GRAD_ACCUM_STEPS = 2` |
| System RAM | 32 GB | 64 GB | peak ~20 GB during shading generation (images held in RAM as grayscale uint8) |
| Disk | 40 GB | 60 GB | docker image + datasets (~1.2 GB) + rolling checkpoints (~2 GB) |
| CPU | 8 cores | 16 cores | data generation is CPU-bound |
| Network | 100 Mbps | — | streams ~50k images from HuggingFace |

Notes:
- The default shading source `imagenet_sketch` streams via HuggingFace
  with no auth. `--source imagenet` needs a HF token with ImageNet
  access; `imagenette`/`caltech101`/`caltech256` need no auth.
- Loaders store images as grayscale uint8 downscaled to the conversion
  canvas, and dataset files store grids as uint8 (the 98-char vocab fits
  in a byte) — full-size RGB float storage would need hundreds of GB.
- Checkpoints are rolling: `{stage}_last.pt` (every epoch) and
  `{stage}_best.pt` (on val improvement), not one file per epoch. One
  generated sample per epoch is archived to `checkpoints/samples/`; the
  Streamlit app's "Training progress" mode scrubs through them.
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
  glyph_sim.py            glyph-similarity soft-target matrix
  text_embed.py           frozen CLIP text encoder for prompt conditioning
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
  streamlit_app.py        web UI (prompted generation + inpainting +
                          training-progress gallery)
tests/                    CPU-fast pytest suite
```

## Generation quality & controllability

Two packages of improvements sit on top of the base model.

**Inference-time (works on any checkpoint, no retraining):**
- **Cosine unmask schedule** (MaskGIT): commit few cells early, many late,
  instead of an even split — early context is poor, so be conservative.
- **Gumbel-annealed confidence**: rank which cells to commit by
  `log P(sampled) + Gumbel noise`, with the noise annealed to zero across
  steps. Explores early, commits greedily late; avoids the deterministic
  error-cascade of raw max-probability ranking.
- **Revision passes**: after the main fill, re-mask the least-confident
  cells and refill them (a cheap Token-Critic-style self-correction). The
  one-pass loop can never fix an early mistake; this can.
- **Vectorized sampling**: one batched `multinomial` over the grid.

**Trained-in (require run #2 with the new data/model):**
- **Text-prompt conditioning + classifier-free guidance (CFG)**: every
  dataset now carries captions — geometry samples describe their
  primitives ("a rectangle and a cross"), ImageNet-Sketch contributes its
  class names, and human ASCII art keeps its titles. Captions are embedded
  once with a frozen CLIP text encoder (`data/text_embed.py`); every
  transformer block **cross-attends to the caption's token sequence**
  (compositional control — grid cells can bind to individual words), and
  the tokens' masked mean feeds a global additive vector. 10% caption
  dropout to a learned null token during training enables CFG, with a
  guidance knob trading diversity for adherence. `generate(...,
  prompt="a small sailboat", guidance_scale=3.0)` (or pass precomputed
  `cond_tokens`/`cond_mask`).
- **Full mask-ratio coverage**: training mask ratios now reach 1.0, so the
  fully-masked grid every generation starts from is in-distribution.
- **Mask-ratio conditioning**: the model is told what fraction of the grid
  is hidden — the denoising "noise level" a plain masked LM never gets.
- **Glyph-aware soft labels** (`data/glyph_sim.py`): confusing `/` with `|`
  costs less than confusing `/` with `@`, because targets are smoothed
  toward *visually similar* glyphs (from rendered-bitmap similarity). The
  model learns the visual structure of the character set.
- **Mixed masking** (`training/masking.py`): random-cell, rectangular-block
  (inpainting) and strided-anchor (upscaling) masks are mixed during
  training, so all three inference modes are in-distribution.
- **EMA weights + bf16 autocast**: a moving average of weights for
  eval/samples/final checkpoint (free quality), and bf16 mixed precision
  (~2x faster on the 4090s).

The conditioning params are **zero-initialized**, so they are a no-op until
trained: a checkpoint from a run without conditioning (e.g. run #1) loads
with `strict=False` and behaves identically, and the inference-time
improvements above apply to it unchanged. The frozen text encoder is only
needed where new text appears (data prep, the start of a conditioned
training stage, interactive generation) — never inside the training loop,
which reads precomputed embeddings of each dataset's unique captions.

## Multi-GPU training (DDP)

Training uses **DistributedDataParallel** when launched with `torchrun`
(or via `python main.py`, which relaunches itself under torchrun when it
detects more than one GPU). How it works:

- **One process per GPU.** torchrun starts N copies of `train.py` and
  gives each a `RANK` (0..N-1). Each process holds a full model replica.
- **Data sharding.** A `DistributedSampler` splits every epoch's samples
  into N disjoint shards, reshuffled per epoch, so no GPU wastes compute
  on a sample another GPU already saw.
- **Gradient all-reduce.** After each backward pass, DDP averages the
  gradients across all replicas (overlapped with the backward computation
  itself), so every optimizer step is identical everywhere and the
  replicas never drift. The effective batch size becomes
  `BATCH_SIZE × num_gpus`, meaning fewer, larger optimizer steps per
  epoch — stage learning rates are automatically multiplied by the GPU
  count to compensate (linear-scaling rule; disable with
  `SCALE_LR_WITH_GPUS = False` in `config.py`).
- **Rank 0 does the talking.** Logging, per-epoch samples, and
  checkpoints come from rank 0 only; checkpoints are saved unwrapped, so
  they load identically with or without DDP.

The DDP path is exercised by a 2-process CPU (gloo backend) smoke test:
sharding, gradient sync (replicas stay bit-identical), loss decrease,
and checkpoint compatibility. On GPUs it uses NCCL automatically. This
model is small (~35M params), so gradient traffic is light and scaling
efficiency should be high; data generation is unaffected (CPU-bound).

## Upscaling (generate beyond 48×80)

Two mechanisms, both free consequences of the architecture:

1. **RoPE extrapolation.** The model has no learned positional
   embeddings — 2D RoPE encodes *relative* row/column offsets inside
   attention. A model trained at 48×80 therefore runs at any grid size up
   to `MAX_ROWS × MAX_COLS` (96×160); quality degrades gracefully at
   relative distances longer than any seen in training. The Streamlit app
   exposes a 64×128 option.
2. **MaskGIT super-resolution** (`upscale_grid` in `model/inference.py`,
   and the "Upscale ×2" button in the app). Take a finished piece, anchor
   each character at position `(2r, 2c)` of a 2× canvas, mask everything
   between, and run the normal iterative unmasking to fill the gaps. The
   anchors are never overwritten, so the result is guaranteed faithful to
   the source — the model only interpolates detail. This is the
   inpainting capability reused as an upscaler; no extra training needed
   (though anchor patterns are out-of-distribution, so expect it to
   improve if you fine-tune with strided-anchor masking).

The RoPE tables are registered as non-persistent buffers, so raising
`MAX_ROWS`/`MAX_COLS` later never invalidates existing checkpoints.

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
3. ~~Better unmasking schedule~~ — **done** (cosine + Gumbel + revision).
4. ~~Vectorize inference sampling~~ — **done**.
5. ~~Conditioning~~ — **done** (text-prompt conditioning + CFG via a frozen
   CLIP text encoder; run #2 trains it. All three stages carry captions —
   geometry primitive descriptions, ImageNet-Sketch class names, human-art
   titles).
6. ~~Cross-attention over caption tokens~~ — **done** (every block
   cross-attends to the caption token sequence; zero-init so old
   checkpoints still load and run).
7. **Larger grids** — RoPE supports up to 96×160; the strided-anchor
   training now makes 2× upscaling in-distribution. Train with variable
   grid sizes to push further.
8. **Image-conditioned rendering** — condition on a source-image encoding
   and train on the converter's own (image, grid) pairs, so the network
   becomes a neural image→ASCII renderer that can beat the deterministic
   converter.
9. **Stronger conditioning** — AdaLN-Zero modulation (DiT-style) instead of
   additive conditioning, if prompt control needs to be sharper.
10. **More human data** — `mrzjy/ascii_art_generation_140k` (139k pieces)
    and `Csplk/THE.ASCII.ART.EMPORIUM` (3.1M rows) as stage-3 expansions.
