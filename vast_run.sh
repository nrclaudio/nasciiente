#!/bin/bash
# One-shot full pipeline for a Vast.ai instance (or any fresh GPU box):
# data generation (geometry + sketch shading + auto-captioned human ASCII
# art + synthetic prompt engine) followed by curriculum training.
#
# Usage (SSH onto the instance, then). The repo is private, so a GitHub
# token with read access is needed (fine-grained PAT, Contents: read):
#   export GH_TOKEN=github_pat_...
#   export HF_TOKEN=hf_...          # recommended: avoids HF rate limits
#   curl -fsSL -H "Authorization: token $GH_TOKEN" \
#     https://raw.githubusercontent.com/nrclaudio/ascii-art-transformer/claude/ascii-art-pr-review-v8c3ok/vast_run.sh | bash
# or clone the repo yourself and run: bash vast_run.sh
#
# Environment overrides:
#   GH_TOKEN=...           GitHub token for cloning the private repo
#   BRANCH=main            git branch to train from
#   SANITY=0               skip the 200-sample sanity pass
#   SYNTH=0                skip the synthetic data engine (train on
#                          sketch data only)
#   SYNTH_SAMPLES=200000   synthetic samples to generate
#   AUTO_CAPTION=0         skip BLIP auto-captioning of human pieces
set -uo pipefail

# Token goes into the remote URL — fine on a throwaway instance, but
# don't reuse a long-lived token here
REPO="https://${GH_TOKEN:+${GH_TOKEN}@}github.com/nrclaudio/ascii-art-transformer"
BRANCH="${BRANCH:-claude/ascii-art-pr-review-v8c3ok}"
WORKDIR="${WORKDIR:-${HOME}/ascii-art-transformer}"
SYNTH="${SYNTH:-1}"
SYNTH_SAMPLES="${SYNTH_SAMPLES:-200000}"
AUTO_CAPTION="${AUTO_CAPTION:-1}"

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# Find a Python. Vast templates activate their venv/conda only in
# interactive shells; `curl | bash` is non-interactive, so do it here.
[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate
PY="$(command -v python || command -v python3)" || {
    echo "ERROR: no python interpreter found on PATH"; exit 1; }
echo "Python: $("$PY" --version 2>&1) at $PY"
echo

# Clone or update
if [ ! -d "$WORKDIR/.git" ]; then
    git clone "$REPO" "$WORKDIR"
fi
cd "$WORKDIR"
git fetch origin "$BRANCH" && git checkout "$BRANCH" && git pull origin "$BRANCH"

echo "=== Installing dependencies ==="
"$PY" -m pip install -q -r requirements.txt || {
    echo "!!! pip install failed"; exit 1; }
if [ "$SYNTH" = "1" ]; then
    "$PY" -m pip install -q diffusers accelerate || {
        echo "!!! pip install diffusers failed"; exit 1; }
fi
echo

# Pre-flight: the text encoder must load and embed before anything long
# runs — a CLIP download/API problem should fail here, in seconds
echo "=== Pre-flight: CLIP text encoder ==="
"$PY" -c "
import sys; sys.path.insert(0, '.')
from data.text_embed import embed_captions
t, m = embed_captions(['a rectangle and a cross'])
print('CLIP text encoder OK:', tuple(t.shape), int(m.sum()), 'tokens')" || {
    echo "!!! Text encoder pre-flight FAILED — fix before burning GPU time."
    exit 1; }
echo

# Sanity pass: convert 200 images and show samples BEFORE the long run,
# so a bad dataset is caught in minutes, not hours
if [ "${SANITY:-1}" = "1" ] && [ ! -f data/shading_data.pt ]; then
    echo "=== Sanity pass: 200 shading samples (eyeball the grids!) ==="
    "$PY" main.py --stage shading --num-shading 200 || {
        echo "!!! Sanity pass FAILED — aborting before the long run."; exit 1; }
    echo
    echo "=== Sanity samples above. Full run starts in 60s — Ctrl-C now"
    echo "=== if the grids look wrong. ==="
    sleep 60
    rm -f data/shading_data.pt   # regenerate at full size below
fi

# --- Data ------------------------------------------------------------
"$PY" main.py --stage geometry --skip-deps
"$PY" main.py --stage shading --skip-deps

if [ ! -f data/human_data.pt ]; then
    echo "=== Human ASCII art ==="
    if [ "$AUTO_CAPTION" = "1" ]; then
        "$PY" data/prepare_human_ascii.py --auto-caption || \
            echo "(human stage optional — continuing)"
    else
        "$PY" data/prepare_human_ascii.py || \
            echo "(human stage optional — continuing)"
    fi
fi

TRAIN_DATA_ARGS=""
if [ "$SYNTH" = "1" ]; then
    if [ ! -f data/synthetic_data.pt ]; then
        echo "=== Synthetic data engine: ${SYNTH_SAMPLES} samples ==="
        "$PY" data/generate_synthetic.py --num-samples "$SYNTH_SAMPLES" \
            --merge data/shading_data.pt || {
            echo "!!! Data engine failed"; exit 1; }
    fi
    TRAIN_DATA_ARGS="--shading-data data/synthetic_data.pt"
fi

# --- Training (all stages; resume variants: --init-from / --stages) ---
"$PY" training/train.py $TRAIN_DATA_ARGS 2>&1 | tee train.log

# Bundle results for easy retrieval before the instance is destroyed
tar czf results.tar.gz checkpoints/final_model.pt checkpoints/samples train.log
echo
echo "=============================================================="
echo "  DONE. Retrieve results before destroying the instance:"
echo "    scp -P <ssh-port> root@<instance-ip>:$WORKDIR/results.tar.gz ."
echo "  (exact command is on the instance card in the Vast console)"
echo "=============================================================="
