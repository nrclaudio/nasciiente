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
- [ ] **ASCII-aware CLIP swap** (roadmap #4) — *plumbing landed*: set
      `RANKER_MODEL` in config.py to the arXiv 2503.08295 released weights'
      HF id (find it via the paper's project page — network was blocked
      from the dev sandbox). Re-ranker uses it automatically; the engine's
      image-side filter stays on vanilla CLIP. Eyeball best-of-4 rankings
      before/after on 10 prompts.
- [ ] **Download ASCIIBench** (5,315 labeled human pieces): stash in backup,
      earmark as stage-3 supplement + eval set.
- [ ] Decide Phase 2 decode defaults from the Halton A/B result.

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

- [ ] **Critic-scored revision / ReMDM-style remasking** (roadmap #5):
      replace the ad-hoc revision passes with scheduled in-loop remasking
      (η knob), scores from the Phase 2 critic head. Expose steps as the
      quality/speed dial in the app.
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
