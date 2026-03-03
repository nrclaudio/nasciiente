#!/bin/bash
#SBATCH --job-name=txt_gen
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo ""

# Activate environment
source /data1/cnovellarausell/miniforge3/etc/profile.d/conda.sh
conda activate /data1/cnovellarausell/envs/ascii

# Move to project directory
cd /data1/cnovellarausell/ascii-art-transformer

# Generate data if not present
if [ ! -f data/geometry_data.pt ]; then
    echo "=== Generating geometry data ==="
    python data/generate_geometry.py
fi

if [ ! -f data/shading_data.pt ]; then
    echo "=== Generating shading data ==="
    python data/generate_shading.py
fi

# Train
echo ""
echo "=== Starting training ==="
python training/train.py

echo ""
echo "=== Done ==="
date
