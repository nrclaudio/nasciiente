#!/bin/bash
# One-shot training run for a Vast.ai instance (or any fresh GPU box).
#
# Usage (SSH onto the instance, then):
#   curl -fsSL https://raw.githubusercontent.com/nrclaudio/ascii-art-transformer/claude/ascii-art-pr-review-v8c3ok/vast_run.sh | bash
# or clone the repo yourself and run: bash vast_run.sh
#
# Environment overrides:
#   BRANCH=main            git branch to train from
#   SANITY=0               skip the 200-sample sanity pass
set -uo pipefail

REPO="https://github.com/nrclaudio/ascii-art-transformer"
BRANCH="${BRANCH:-claude/ascii-art-pr-review-v8c3ok}"
WORKDIR="${WORKDIR:-${HOME}/ascii-art-transformer}"

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

# Sanity pass: convert 200 images and show samples BEFORE the long run,
# so a bad dataset is caught in minutes, not hours
if [ "${SANITY:-1}" = "1" ]; then
    echo "=== Sanity pass: 200 shading samples (eyeball the grids!) ==="
    "$PY" main.py --stage shading --num-shading 200 || {
        echo "!!! Sanity pass FAILED — aborting before the long run."; exit 1; }
    echo
    echo "=== Sanity samples above. Full run starts in 60s — Ctrl-C now"
    echo "=== if the grids look wrong. ==="
    sleep 60
    rm -f data/shading_data.pt   # regenerate at full size below
fi

# Full pipeline: data generation + all training stages.
# main.py installs deps, uses all GPUs (torchrun/DDP) automatically.
"$PY" main.py 2>&1 | tee train.log

# Bundle results for easy retrieval before the instance is destroyed
tar czf results.tar.gz checkpoints/final_model.pt train.log
echo
echo "=============================================================="
echo "  DONE. Retrieve results before destroying the instance:"
echo "    scp -P <ssh-port> root@<instance-ip>:$WORKDIR/results.tar.gz ."
echo "  (exact command is on the instance card in the Vast console)"
echo "=============================================================="
