# STATUS — nASCIIente (repo: nasciiente)

*Last updated: 2026-07-14. Written so the project can be picked up cold —
by future-you or anyone else — after any amount of time away.*

**Update 2026-07-14:** a literature deep-dive produced
[`docs/research-roadmap.md`](docs/research-roadmap.md) (ranked findings)
and [`TODO.md`](TODO.md) (phased execution list); every code-side item is
implemented, tested (128 passing), and flag-gated — see TODO for what's
landed vs what awaits a GPU run. New decode flags on existing
checkpoints: `--order halton`, `--remask-eta`, `--critic-conf` (needs
retrain), plus training-side critic head / rare-caption upsampling /
30-epoch shading, engine quality tags + sparse dialect routing, and
`data/prepare_asciibench.py` (4,860 labeled human pieces).

## What this is

A 34.5M-parameter bidirectional transformer (ASCIIBert) that turns a text
prompt ("a dragon") into a 48×80 grid of ASCII characters, trained from
scratch on a self-manufactured captioned dataset (FLUX-generated images →
deterministic glyph conversion, three style dialects per image). Decoding
is MaskGIT-style iterative unmasking. Full first-principles documentation
lives in [`docs/study-guide.md`](docs/study-guide.md) — architecture,
training recipe, data engine, and all nine documented failures with their
fixes. Read that before touching anything; this file is only the bookmark.

## State at a glance

| Component | Status |
|---|---|
| Model architecture (2D RoPE, cross-attn, zero-init splice) | done, stable |
| Data engine v3 (FLUX, 3 dialects/image, composites) | done, working |
| Training recipe (dropout 0, auto space weight, self-context v2) | done, validated |
| Curriculum (geometry → shading → human, multi-source replay) | done; replay verified at shading_last |
| Inpainting / upscaling | working (diamond inpaint: 100% acc/prec/recall) |
| Edge echo (nested duplicate edges) | **solved** — root-caused and fixed, see below |
| Geometry free generation | clean with capped decode (perfect cross, closed triangle) |
| Tonal free generation (dense subjects: bonsai, silhouettes) | working |
| Tonal free generation (sparse subjects: dragon, lighthouse) | works uncapped (865-ink winged dragon); hybrid tail-cap decode pushed, **final validation pending** |
| Serving (FastAPI + Streamlit, best-of-k CLIP re-rank) | working; decode defaults wired in |

## The one pending validation

The hybrid decode (`bf91eb5`) runs the head uncapped (layout forms under
gumbel exploration + rising guidance) and caps only the tail below mask
ratio 0.35 (where the ambiguous cells mass-commit and echo is born). It
should give the winged dragon AND the clean rectangle under one setting.
Confirm with:

```bash
python probe.py --checkpoint checkpoints/shading_last.pt \
  --prompt "a dragon" --prompt "a lighthouse" --prompt "a bonsai tree" \
  --prompt "a dragon, outline style" --prompt "a rectangle" \
  --guidance 1.5 --schedule rise --gumbel 1 --steps 32 \
  --max-commit 8 --revision-steps 0 --out probe_hybrid.txt
```

If the dragon is coherent and the rectangle is clean in the same file, the
canonical decode is settled:
**guidance 1.5 · schedule rise · gumbel 1 · steps 32 · max-commit 8 ·
cap-below 0.35** (already the config/app defaults; use gumbel 0 for
maximum-precision geometry). Then probe `final_model.pt` (add `--ema`)
once the human stage finishes, and ship.

Secondary open item: "a lighthouse" was mediocre even uncapped on the
latest shading checkpoint (the older run drew it better). Could be seed
luck — best-of-k at serve time is the intended mitigation — or a mild
regression worth one A/B against the previous checkpoint in the backup.

## What the last debugging cycle established (July 2026)

Full detail in the study guide (§14, Appendix C). Headlines, with commits:

- **Dropout 0.1 silently blocked prompt following** (`9fba787`) — proven
  by a 64-grid memorization ablation (CE 1.0 vs 0.034).
- **Space loss weight is now auto-calibrated** per dataset (`97aeb28`).
- **Self-context training** closes exposure bias; the loss MUST cover the
  self-committed cells or it teaches the opposite lesson (`334e8ac` —
  probe-verified in both directions).
- **Edge echo = parallel-commit multimodality in the schedule tail**;
  fixed by the commit cap (`a83fc47`), then scoped to the tail only after
  full capping blank-collapsed sparse tonal prompts (`bf91eb5`). The
  rise-guidance schedule is keyed to step progress, not mask ratio
  (`faf57ef`).
- **`draw_triangle` never drew a closed triangle** (`fe65ff6`) — 200k
  mislabeled samples; regenerate geometry data before any retrain.

## How to resume cold

1. `git clone` this repo, branch `claude/ascii-art-pr-review-v8c3ok`
   (or wherever it has been merged since).
2. Heavy artifacts are NOT in git. Checkpoints and datasets live in:
   - laptop backup: `~/glyph48-backup/` (rsync of the full training
     instance, checkpoints + data included)
   - Hugging Face: private repo `glyph48-artifacts` (checkpoints, *.pt)
3. To retrain from nothing instead: `python data/generate_geometry.py`
   (regenerates geometry with the fixed triangle), regenerate synthetic
   data with `data/generate_synthetic.py` (FLUX; see study guide App. B),
   then `python -u training/train.py --shading-data data/synthetic_v3.pt`.
4. Probes: `probe.py` (free generation), `probe_inpaint.py` (structure
   knowledge, scored), `probe_coarse.py` (coarse-to-fine; deprioritized —
   the tail-capped decode beat it). All accept `--max-commit/--cap-below`.
5. Serve: `uvicorn app.server:app` or `streamlit run app/streamlit_app.py`.
6. Tests: `python -m pytest tests/ -q` (125 passing at last update).

## If shelving

Backups above are complete as of 2026-07-13. Destroy the GPU instance
(it bills hourly), revoke the instance's GitHub fine-grained token and
any write-scoped HF token. Nothing else leaks money or secrets; the
project resumes from step 1 above whenever.
