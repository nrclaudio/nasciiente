# Research Roadmap — what would help GLYPH48 most

*Compiled 2026-07-13 from a 20-source literature sweep (92 extracted claims;
8 adversarially verified 3-0, the rest quote-grounded but unverified — the
verification fleet hit an API budget mid-run) crossed against a 4-agent
audit of this codebase's extension points. Ranked by expected quality gain ×
implementation fit ÷ cost, for a solo developer on one rented GPU with CPU
inference.*

---

## The headline

The literature agrees with what our probes found the hard way: for masked
parallel decoders, **the biggest remaining gains are in the sampler and the
training budget, not the architecture**. Three independent lines of work
(Halton scheduling, Token-Critic, ReMDM) all attack exactly the two
pathologies we diagnosed by hand — clustered parallel commits and
frozen-once-committed errors — and all report large quality gains on
ImageNet-scale token grids with *no* generator retraining or minor
fine-tuning. Meanwhile the data-scaling literature says our training runs
are likely 10–50× too short for a masked model on fixed data, and the one
ASCII-specific deep-learning paper found that **vanilla CLIP scores ASCII
art at chance level** — which quietly undermines three of our subsystems
(best-of-k re-ranking, the data engine's consistency filter, and any
CLIP-based eval).

---

## Tier 1 — do these first

### 1. Train much longer (and upsample rare subjects)

**Evidence.** A data-constrained-scaling study of masked-diffusion vs
autoregressive models (7M–2.5B params, bracketing our 34.5M) found masked
models tolerate **~100+ epochs of data repetition** before repeated data
loses value (data-reuse half-life ~500 vs ~15 for AR), reach best
validation loss around ~500 epochs vs ~50 for AR, and showed no overfitting
within budget — because random masking is itself data augmentation: every
epoch sees each sample through a fresh mask. Separately, knowledge-capacity
scaling work (arXiv 2404.05405) finds a piece of knowledge needs **~1000
training exposures** to be stored at full fidelity; at ~100 exposures,
capacity halves. *(Both unverified-by-panel; multiple corroborating claims
extracted.)*

**Why it fits us.** We train 6 epochs per stage. Our "geometry flatlines by
epoch 2" observation drove epochs down — but geometry is trivially
memorizable; the 500k-sample FLUX shading stage almost certainly is not.
Sparse subjects — our weakest regime — are exactly the under-exposed tail
the 1000-exposure rule predicts. The masking curriculum (fresh random mask
per epoch, already our only regularizer) is the mechanism the literature
credits for repetition tolerance.

**Action.** Raise `SHADING_EPOCHS` from 6 to 30–60 with the cosine anneal
stretched over the full run; track val loss for the (unlikely) overfit
signal. Add per-subject upsampling so rare/sparse subjects get several
times more exposures. Cost: pure GPU-hours, zero code risk. **This is the
cheapest large lever we have.**

### 2. Token-Critic head — the training signal is already computed

**Evidence (3-0 verified).** Token-Critic (arXiv 2209.04439) trains an
auxiliary model to distinguish original from generator-sampled tokens in a
masked-and-reconstructed image; at decode its scores replace the
generator's own confidence for keep-vs-remask decisions. ImageNet 256
FID 6.56 → 4.69 over the MaskGIT baseline. It also works as a **post-hoc
repair pass**: re-masking the worst 60% of a *finished* sample by critic
score improved FID/IS from 8.48/167 to 7.64/182. Two more papers (Google's
discrete predictor-corrector; the "informed corrector" of arXiv 2407.21243)
independently confirm learned correctors beat generator-confidence
heuristics for exactly the compounding-parallel-error failure class.
*(Corroborating claims unverified-by-panel.)*

**Why it fits us.** This is the load-bearing finding of the codebase audit:
`self_context_corrupt` already produces grids containing the model's own
commits, and computes — then **discards** — the `reveal` mask and the
labels (`sampled != target`) that are precisely a critic's training data
(`train.py:253-255`). Our confidence function is consumed at exactly two
call sites (`inference.py:209, 343` — commit ranking and revision
re-masking), so a trained critic drops in without decode surgery. A
zero-init `Linear(D→1)` head after the final LN (~500 params, following
the repo's established zero-init + tolerant-checkpoint retrofit pattern)
loads onto existing checkpoints and fine-tunes cheaply. Our "poor-man's
Token-Critic" revision passes — which we measured doing nothing before
self-context training — become the real thing, with a signal trained for
the job.

**Action.** ~1–2 days train side, hours decode side. Also enables the
critic-driven repair pass as a cheap post-processing knob.

### 3. Halton (low-discrepancy) commit scheduler — training-free

**Evidence (3-0 verified).** arXiv 2503.17076 analyzes MaskGIT's
confidence-ordered unmasking via mutual information: confidence-first
selection commits **spatially clustered, highly correlated tokens
together**, yielding low information gain per step and non-recoverable
errors. Their fix — selecting commit *positions* by a quasi-random Halton
sequence that spreads commits uniformly across the 2D grid — is a
**training-free drop-in** that improved ImageNet 512 FID from 8.38 to 6.11
at 32 steps.

**Why it fits us.** Their MI analysis is a formal derivation of the exact
two pathologies we found empirically: (a) the edge echo (clustered
correlated commits = several offset copies of the same edge), and (b) the
blank cascade (greedy confidence = space-first commits reinforcing
blankness). A position-spread scheduler attacks both at once — it never
commits a whole edge's worth of correlated cells in one step, and it never
commits only the safest (space) cells either. It may **replace both the
commit cap and the space-bias knob with one simpler mechanism**, and
possibly resolve the sparse-tonal collapse that the cap introduced.

**Action.** Implement position-spread selection as a third scheduler
option in `_iterative_fill` (`--order halton|confidence`), A/B against the
current hybrid tail-cap decode on the standard probe set. Days, no
retraining, fully reversible.

### 4. ASCII-aware CLIP — three subsystems are running on a broken metric

**Evidence.** "ASCIIBench: Evaluating Language-Model-Based Understanding
of Visually-Oriented Text" (arXiv 2512.04125) measured off-the-shelf CLIP
on ASCII art
and found cosine similarity **cannot separate most ASCII categories —
chance-level performance** — attributing the failure to the representation
itself. The authors release (a) fine-tuned ASCII-aware CLIP weights and
(b) **ASCIIBench**, 5,315 human-made, class-labeled ASCII pieces.
*(Unverified-by-panel; quote-grounded.)*

**Why it fits us.** Three of our subsystems assume CLIP understands
rendered ASCII: the best-of-k re-ranker, the data engine's 0.2-cosine
consistency filter, and any future preference/eval signal. If vanilla CLIP
is near chance on ASCII structure, our re-ranker is picking best-of-k
nearly at random and the engine filter is weaker than believed. (Note: the
engine filter scores the *source image*, not the ASCII — that use is
probably fine; the re-ranker scores *rendered grids* — that one is
directly implicated.)

**Action.** Swap the released ASCII-CLIP into `clip_rank.py` (hours; it's
one model id if the weights are HF-hosted), A/B re-ranking quality by eye
on best-of-4 batches. Adopt ASCIIBench as (a) an eval set and (b) 5.3k
extra human pieces for the stage-3 fine-tune. Keep vanilla CLIP for the
engine's image-side filter.

---

## Tier 2 — decode upgrades after Tier 1 lands

### 5. ReMDM-style scheduled remasking (replaces ad-hoc revision passes)

**Evidence (partially 3-0 verified).** ReMDM (arXiv 2503.00307) derives a
principled remasking backward process applicable to *pretrained* masked
models with no retraining. On ImageNet 256 token grids over a pretrained
MaskGIT it beat the confidence sampler (FID 4.45 vs 4.85 at T=64) and —
notably — exhibited genuine **inference-time compute scaling**: quality
keeps improving with more steps where vanilla masked decoding saturates.
Ships as a family of schedules (cap/rescale/conf/loop) with one intensity
knob η.

**Why it fits us.** Our revision passes are a hand-rolled version of this
(re-mask k least-confident, refill). ReMDM integrates remasking *into* the
step loop with a principled schedule — and pairs naturally with a critic
(#2) supplying the remasking scores. For a CPU-deployable model, "more
steps = more quality" is exactly the knob we want to expose.

### 6. Self-guidance fine-tune (bigger than CFG, cheaper than it looks)

**Evidence.** A NeurIPS 2024 paper attributes masked models' quality gap
vs diffusion to missing guidance mechanisms and proposes *self-guidance*:
contrast the model's prediction against a "semantically smoothed"
auxiliary prediction (discrete analog of blur guidance). Roughly **halves
MaskGIT's FID (6.56 → 3.35) at 24 NFEs**, via parameter-efficient
fine-tuning only, claimed better quality-diversity than Token-Critic at
lower cost. *(Unverified-by-panel.)* Their caution mirrors our CFG
finding: guidance scale >3 over-sharpens — the textbook-knob lesson again.

**Why it fits us.** Competes with #2 for the same budget; the literature
disagrees about which wins, so run the cheap one first (#2 — its training
signal is free here) and try self-guidance if coherence still lags. Our
glyph-cluster machinery is a ready-made "semantic smoothing" kernel —
smoothing logits over visually-similar glyphs is a one-liner with
`glyph_sim`.

### 7. Logit self-conditioning (~1 day, RIN's key trick without RIN)

**Evidence.** RIN (arXiv 2212.11972) showed latent self-conditioning —
feeding the previous iteration's state into the current forward pass — is
the mechanism that makes iterative generation with routing work. The full
RIN architecture is a poor fit (we have no latent bottleneck and a
CPU-inference budget), but self-conditioning transfers: condition each
decode step on the previous step's *probabilities*, not just its commits.

**Why it fits us.** The audit located the attach point: a zero-init
`Linear(V→D)` on previous-step probs in `CombinedEmbedding`, with
`_iterative_fill` already carrying per-step probs. Train it during
self-context batches (which already produce a "previous prediction" for
free from the no-grad forward). Gives every step a memory of the full
soft hypothesis space, not just the hard commits — directly aimed at
global-shape coherence.

---

## Tier 3 — data engine (aims squarely at the sparse-tonal weakness)

### 8. Structure-first rendering for sparse subjects (the 2010 ASCII paper called it)

**Evidence.** The classic structure-based ASCII art paper (Xu, Zhang &
Wong, SIGGRAPH 2010) states that tone-based ASCII **fundamentally requires
high text resolution** — tone variety scales with resolution — so at low
resolution, line-structure rendering is the legible strategy, and content
should be *deformed to snap onto the character grid* rather than sampled
per-cell. *(Unverified-by-panel; this is the field's foundational paper.)*

**Why it fits us.** This reframes our "sparse-subject tonal weakness" as
partially a **representational limit, not a model failure**: at 48×80, a
sparse tonal subject may simply not have enough cells to carry tone. Which
matches observation — our outline-dialect dragon is gorgeous while the
tonal dragon struggles. Three concrete moves: (a) route thin/sparse
subjects to the outline dialect in the engine (make outline the *default*
for subjects below an ink threshold, tag tonal explicitly for them);
(b) consider a snap-to-grid deformation stage in the converter (AISS-style)
so training targets have grid-aligned strokes; (c) offer larger canvases
(96×160 upscale) for tonal subjects — RoPE already extrapolates.

### 9. Quality-tier and dialect source tags

**Evidence.** The knowledge-capacity work (arXiv 2404.05405) found mixing
lower-quality data can cost up to **20× of usable capacity — and that
prepending a source/domain token largely restores it**. *(Unverified-by-
panel.)*

**Why it fits us.** We already tag dialects in captions (the mechanism
works — dialect control is probe-verified). Extend the same trick to
quality: tag engine samples by filter margin (e.g. high/low CLIP
consistency tier) so the model can *condition on* quality instead of
averaging over it — and serve with the high-quality tag at generation
time. Hours in the engine; free at inference.

---

## Tier 4 — bigger bets, only if coherence still lags after Tiers 1–3

- **Trained coarse-to-fine (VAR/HMAR/SRDD family).** Next-scale prediction
  produced the largest quality jumps in the 2024–25 image-AR literature
  (VAR: FID 18.65 → 1.73; both 3-0-style headline claims here are
  unverified-by-panel), and HMAR shows masked prediction can be *added to*
  a scale-wise model by fine-tuning, with block-sparse attention cutting
  cost. Our zero-training coarse-to-fine experiment failed, but the
  literature version **trains** the scale conditioning. The blocker is
  token semantics: ASCII glyphs don't downsample like VQ latents — a
  coarse scale would need its own charset-native definition (e.g. the
  anchor-lattice as the coarse scale — machinery we already train).
  Substantial retooling; park it.
- **MAR-style per-token diffusion head** (arXiv 2406.11838): compelling for
  large/continuous vocabularies; our 98-glyph categorical head is not the
  bottleneck. Skip.
- **D3PO preference tuning** (arXiv 2503.08295; DPO for masked discrete
  diffusion — closed
  form exactly in our corruption family): the theory fits perfectly and we
  log best-of-k picks already, but published validation is toy-scale.
  Revisit in six months.
- **Architecture swaps (SwiGLU, gated MLPs, more depth):** the capacity
  literature actively supports our vanilla-transformer-plus-RoPE design at
  short training budgets, and our memorization test showed capacity is not
  binding. Don't spend here.

---

## Suggested sequence

1. **This week (no retrain):** Halton scheduler A/B (#3) + ASCII-CLIP
   re-ranker swap (#4). Both training-free, both measurable with the
   standard probe set in a day each.
2. **Next training run:** longer shading stage with rare-subject
   upsampling (#1) + critic head trained from the already-computed
   self-context signal (#2) + quality tags (#9) + sparse-subject dialect
   routing (#8). One run carries all four.
3. **After that run's probes:** ReMDM-style remasking with critic scores
   (#5), then self-guidance (#6) or self-conditioning (#7) if coherence
   still lags.

## Verification status & sources

8 claims verified by 3-0 adversarial panels (Halton ×4, Token-Critic ×3,
ReMDM ×1); 17 more entered verification but the verifier fleet hit an API
budget limit (0 claims were refuted — nothing surveyed was debunked);
the remainder are quote-grounded extractions from primary sources.
Treat unverified numbers as "reported by the paper," not independently
checked.

Primary sources: arXiv 2503.17076 (Halton scheduler) · 2209.04439
(Token-Critic) · 2503.00307 (ReMDM) · NeurIPS 2024 self-guidance for MGMs ·
Google Research discrete predictor-corrector · 2407.21243 (informed
corrector) · VAR (OpenReview gojL67CfS8) · 2506.04421 (HMAR) · SRDD ·
2406.11838 (MAR/Diffusion Loss) · 2212.11972 (RIN) · discrete flow
matching · masked-diffusion-vs-AR data-constrained scaling · 2305.16264
(data-constrained LM scaling) · 2404.05405 (knowledge capacity) ·
2503.08295 (D3PO) · Xu/Zhang/Wong structure-based ASCII art (SIGGRAPH
Asia 2010) · 2512.04125 (ASCIIBench + ASCII-aware CLIP) · 2503.14375
(ASCII tile-classification study).
