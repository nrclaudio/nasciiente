# TODO — GLYPH48 execution list

*Ordered by dependency and payoff-per-effort. Derived from STATUS.md (in-flight
work) + docs/research-roadmap.md (literature findings). Each phase gates the
next; nothing in Phase 2 starts until Phase 1's probes are read.*

## Phase 0 — close out what's in flight *(half a day, no training)*

- [ ] **Validate the hybrid tail-cap decode** on `shading_last`:
      `python probe.py --checkpoint checkpoints/shading_last.pt --prompt "a dragon" --prompt "a lighthouse" --prompt "a bonsai tree" --prompt "a dragon, outline style" --prompt "a rectangle" --guidance 1.5 --schedule rise --gumbel 1 --steps 32 --max-commit 8 --revision-steps 0 --out probe_hybrid.txt`
      Pass = coherent dragon AND clean rectangle in one file.
- [ ] **Probe `final_model.pt`** (once the human stage finishes) with the same
      settings plus `--ema`; check geometry replay survived stage 3.
- [ ] **Backup sweep**: rsync to laptop + `hf upload` checkpoints; verify
      timestamps.

## Phase 1 — zero-training experiments *(2–3 days, existing checkpoints)*

- [x] **Halton commit scheduler — CODE LANDED, run pending** (roadmap #3,
      verified 3-0, training-free). `--order halton` in probe.py; verified
      mechanically (true permutation, spatial spread, order honored,
      works with cap/inpaint/revision). RUN: A/B against the hybrid cap
      decode on the standard probe set (rectangle/diamond/cross/triangle +
      dragon/lighthouse/bonsai). If it wins BOTH regimes it replaces the
      cap AND the space-bias — simpler decode, delete knobs.
- [ ] **ASCII-aware CLIP** (roadmap #4) — *plumbing landed, weights NOT
      yet public*: github.com/KerryLuo/ASCIIBench is "under construction"
      and ships only the dataset — the released-weights claim from the
      research sweep was one of the unverified ones and didn't hold (yet).
      Options: (a) email the authors (contact in their README) asking for
      the fine-tuned CLIP; (b) fine-tune our own on rendered ASCIIBench
      pieces (752 classes = contrastive labels; a small LoRA job); (c)
      wait. `RANKER_MODEL` in config.py takes the HF id whenever one
      exists.
- [x] **ASCIIBench dataset — INGESTED**: `data/prepare_asciibench.py`
      converts final_dataset.jsonl -> training payload (verified on the
      real download: 4,860 pieces kept, 736 classes, 18.7 MB). Use as
      stage-3 supplement/replay + eval set. License unstated upstream —
      keep the data out of the public repo; regenerate with:
      `curl -sLO https://raw.githubusercontent.com/KerryLuo/ASCIIBench/main/final_dataset.jsonl && python data/prepare_asciibench.py --jsonl final_dataset.jsonl`
- [x] **ReMDM-style in-loop remasking — CODE LANDED, run pending**
      (roadmap #5; inference-only per the paper, so it belongs in this
      phase, not Phase 3). `--remask-eta 0.1-0.5` in probe.py: each step
      re-masks the least-confident committed cells at an annealing
      fraction, keeping early commitments revisable while layout forms —
      the principled replacement for `--revision-steps` (use one, not
      both). Verified mechanically; eta=0 is bit-identical to before.
      RUN: A/B eta 0 / 0.2 / 0.5 alongside the Halton A/B.
- [ ] Decide Phase 2 decode defaults from the Halton × remask-eta A/B
      grid.

## Phase 2 — the next training run (one run carries five upgrades) *(prep ~3–4 days, then GPU time)*

- [ ] **Regenerate geometry data** (`python data/generate_geometry.py`) —
      picks up the fixed closed triangle. Required before any retrain.
- [x] **Longer shading stage — CODE LANDED** (roadmap #1): `SHADING_EPOCHS`
      = 30 in config. RUN: watch val loss per epoch for the (unlikely)
      overfit signal; budget GPU rental ~5× the old stage cost.
- [x] **Rare-subject upsampling — CODE LANDED**: `upsample_rare()` in
      train.py duplicates below-median-count captions up to
      `RARE_UPSAMPLE_MAX_FACTOR` (5x), DDP-safe, unit-tested.
- [x] **Token-Critic head — CODE LANDED** (roadmap #2, verified 3-0):
      zero-init `critic` Linear in ASCIIBert (tolerant checkpoint loading
      tested); BCE trained every batch from visible-cell correctness
      (negatives supplied free by self-context reveals);
      `CRITIC_LOSS_WEIGHT = 0.1`. Decode: `--critic-conf` in probe.py
      ranks commits AND revision re-masks by the trained head. RUN: only
      meaningful on a checkpoint trained with the critic on.
- [x] **Quality-tier caption tags — CODE LANDED** (roadmap #9): engine
      appends ", clean render" to samples whose CLIP-to-own-caption score
      ≥ `QUALITY_TIER_CLIP` (0.28); tested with a fake pipe. Serve with
      the tag at inference for the top tier.
- [x] **Sparse-subject dialect routing — CODE LANDED** (roadmap #8): in
      `--mix all`, subjects whose tonal conversion inks < `SPARSE_TONAL_INK`
      (0.08) hand the untagged caption to their outline variant; tonal
      becomes ", shaded" for them. Tested with a fake pipe.
- [ ] Launch the run; probe geometry_last and shading_last with BOTH decode
      variants (hybrid cap vs Halton) as each stage lands.

## Phase 3 — decode upgrades on the new checkpoint *(1–2 weeks, ordered by probe results)*

- [ ] **Critic-scored remasking** (roadmap #5, second half): the in-loop
      remasking is implemented (Phase 1) with generator-confidence
      scores; once the Phase-2 critic exists, re-score the remask
      selection with it and expose decode steps as the quality/speed
      dial in the app.
- [ ] **Critic repair pass** as a serve-time option: re-mask worst-N% by
      critic on finished grids, refill (the Token-Critic paper's post-hoc
      trick).
- [ ] Only if global coherence still lags: **self-guidance PEFT fine-tune**
      (roadmap #6 — use `glyph_sim` clusters as the smoothing kernel) or
      **logit self-conditioning** (roadmap #7 — zero-init `Linear(V→D)` in
      `CombinedEmbedding`, trained during self-context batches). Pick ONE,
      A/B, keep the winner.

## Phase 4 — ship

- [ ] Lock decode defaults from Phase 3 probes; update `config.py`, server,
      Streamlit defaults.
- [ ] Full curriculum probe sweep on `final_model.pt` (+ `--ema`), all
      dialects + geometry replay check + inpaint/upscale sanity.
- [ ] Update STATUS.md and the study guide (new failures/fixes if any;
      flip roadmap items to done).
- [ ] Backup sweep; revoke instance tokens if pausing.

## Parked (revisit only if Phases 1–3 don't close the gap)

- Trained coarse-to-fine / next-scale (VAR/HMAR family) — needs a
  charset-native coarse scale (anchor lattice?); substantial retooling.
- D3PO preference tuning on best-of-k picks — theory fits our corruption
  family exactly, published validation still toy-scale.
- Bigger model / architecture swaps — capacity is measurably not the
  bottleneck; the literature endorses the current vanilla+RoPE design.

---

*Rule of thumb throughout: every change lands behind a flag, every claim
gets a probe, and nothing merges without an A/B at fixed seed against the
standard probe set.*
