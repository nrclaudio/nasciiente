#!/usr/bin/env python3
"""
ASCII Art Transformer — single entry point for Vast.ai GPU instances.

Usage:
    python main.py                        # full pipeline (generate + train)
    python main.py --stage generate       # generate both datasets only
    python main.py --stage train          # train only (data must exist)
    python main.py --stage geometry       # generate geometry data only
    python main.py --stage shading        # generate shading data only
    python main.py --source imagenette    # use Imagenette instead of ImageNet
    python main.py --num-shading 50000    # fewer shading samples
"""

import argparse
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Resolve project root so all imports work regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def print_banner():
    print("=" * 64)
    print("  ASCII Art Transformer")
    print("  MaskGIT-style iterative generation with 2D RoPE")
    print("=" * 64)


def print_system_info():
    """Print GPU/system diagnostics."""
    import torch

    print("\n--- System Info ---")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  PyTorch:   {torch.__version__}")
    print(f"  CUDA:      {torch.version.cuda or 'N/A'}")
    print(f"  Device:    {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU:       {props.name}")
        print(f"  VRAM:      {props.total_mem / 1e9:.1f} GB")
        print(f"  SM count:  {props.multi_processor_count}")
    else:
        print("  WARNING: No CUDA GPU detected — training will be very slow")

    print(f"  Project:   {PROJECT_ROOT}")
    print()


def install_deps():
    """Install Python dependencies if missing."""
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_path):
        return

    # Quick check: try importing the heaviest optional dep
    try:
        import datasets  # noqa: F401
        return  # already installed
    except ImportError:
        pass

    print("Installing dependencies...")
    # Prefer uv (available on Vast.ai template) for speed, fall back to pip
    uv = subprocess.run(["which", "uv"], capture_output=True)
    if uv.returncode == 0:
        cmd = ["uv", "pip", "install", "-r", req_path]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "-r", req_path]

    subprocess.run(cmd, check=True)
    print()


# ---------------------------------------------------------------------------
# Stage runners — thin wrappers that call into existing modules
# ---------------------------------------------------------------------------

def run_generate_geometry():
    """Generate synthetic geometry data (Stage 1)."""
    from data.generate_geometry import main as geo_main

    out_path = os.path.join(PROJECT_ROOT, "data", "geometry_data.pt")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Geometry data already exists ({size_mb:.0f} MB): {out_path}")
        print("  Delete it to regenerate. Skipping.\n")
        return

    print("\n>>> Generating geometry data...\n")
    geo_main()
    print()


def run_generate_shading(source, num_samples):
    """Generate image-derived shading data (Stage 2)."""
    from config import SHADING_NUM_SAMPLES
    from data.generate_shading import (
        SOURCES,
        compute_shape_vectors,
        image_to_ascii_grid,
        load_images,
        _make_circle_masks,
    )
    from data.charset import idx_to_char
    import torch
    import numpy as np

    out_path = os.path.join(PROJECT_ROOT, "data", "shading_data.pt")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Shading data already exists ({size_mb:.0f} MB): {out_path}")
        print("  Delete it to regenerate. Skipping.\n")
        return

    num_samples = num_samples or SHADING_NUM_SAMPLES

    print(f"\n>>> Generating shading data ({source}: {SOURCES[source]})...\n")

    # Build shape vectors
    print("Building 6D shape vectors...")
    masks = _make_circle_masks()
    mask_sums = np.array([m.sum() for m in masks], dtype=np.float32)
    shape_vectors, char_indices = compute_shape_vectors(masks)
    sv_sq_sum = (shape_vectors ** 2).sum(axis=1, keepdims=True).T

    from config import GRID_H, GRID_W
    from data.generate_shading import (
        CELL_H, CELL_W, NUM_CHARS, AG_KERNEL, CLAHE_CLIP,
        CLAHE_TILE_H, CLAHE_TILE_W, EDGE_BLEND, EDGE_BOOST, CONTRAST_POWER,
    )
    print(f"Grid: {GRID_H}x{GRID_W}, Cell: {CELL_H}x{CELL_W}")
    print(f"Source: {source} — {SOURCES[source]}")
    print(f"Method: 6D shape vector matching ({NUM_CHARS} chars)")

    # Load images
    data_dir = os.path.join(PROJECT_ROOT, "data")
    all_images = load_images(source, data_dir, num_samples)
    if not all_images:
        print("ERROR: No images loaded. Check dataset source and auth.")
        sys.exit(1)

    total_available = len(all_images)
    if total_available < num_samples:
        print(f"  {total_available:,} images available, will cycle to {num_samples:,}")

    # Generate samples
    print(f"\nGenerating {num_samples:,} shading samples...")
    samples = []
    t_gen = time.time()
    log_interval = max(1, num_samples // 20)

    for i in range(num_samples):
        img = all_images[i % total_available]
        if i >= total_available and torch.rand(1).item() > 0.5:
            img = img.flip(-1)
        samples.append(image_to_ascii_grid(
            img, shape_vectors, masks, char_indices, mask_sums, sv_sq_sum))
        if (i + 1) % log_interval == 0:
            elapsed = time.time() - t_gen
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate
            print(f"  {i+1:>7,}/{num_samples:,}  "
                  f"({100*(i+1)/num_samples:5.1f}%)  "
                  f"{rate:.0f} img/s  ETA {eta:.0f}s")

    gen_elapsed = time.time() - t_gen
    print(f"Done in {gen_elapsed:.1f}s ({num_samples/gen_elapsed:.0f} img/s)")

    data = torch.stack(samples)
    print(f"Tensor: {data.shape}, dtype: {data.dtype}")

    # Stats
    unique, counts = torch.unique(data, return_counts=True)
    top_k = counts.argsort(descending=True)[:10]
    print(f"Character usage ({len(unique)} unique):")
    for idx in top_k:
        ch = idx_to_char(unique[idx].item())
        pct = 100 * counts[idx].item() / data.numel()
        print(f"  '{ch}' (idx {unique[idx].item():>2}): {pct:.1f}%")

    torch.save(data, out_path)
    print(f"Saved to {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

    # Print a few samples
    from data.charset import grid_to_string
    print("\n=== Sample Shading Grids ===\n")
    for i in range(min(3, len(samples))):
        print(f"--- Sample {i} ---")
        print(grid_to_string(samples[i]))
        print()


def run_train():
    """Run two-stage training."""
    from training.train import main as train_main

    geo_path = os.path.join(PROJECT_ROOT, "data", "geometry_data.pt")
    shading_path = os.path.join(PROJECT_ROOT, "data", "shading_data.pt")

    if not os.path.exists(geo_path):
        print(f"ERROR: Geometry data not found at {geo_path}")
        print("  Run: python main.py --stage geometry")
        sys.exit(1)
    if not os.path.exists(shading_path):
        print(f"ERROR: Shading data not found at {shading_path}")
        print("  Run: python main.py --stage shading")
        sys.exit(1)

    print("\n>>> Starting two-stage training...\n")
    train_main()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="ASCII Art Transformer — full pipeline for Vast.ai GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "generate", "geometry", "shading", "train"],
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--source", default="imagenet",
        choices=["imagenet", "imagenette", "caltech101", "caltech256", "stl10"],
        help="Image source for shading data (default: imagenet)",
    )
    parser.add_argument(
        "--num-shading", type=int, default=None,
        help="Number of shading samples (default: from config)",
    )
    parser.add_argument(
        "--skip-deps", action="store_true",
        help="Skip dependency installation check",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print_banner()

    if not args.skip_deps:
        install_deps()

    print_system_info()

    t0 = time.time()

    if args.stage in ("all", "generate", "geometry"):
        run_generate_geometry()

    if args.stage in ("all", "generate", "shading"):
        run_generate_shading(args.source, args.num_shading)

    if args.stage in ("all", "train"):
        run_train()

    elapsed = time.time() - t0
    print("=" * 64)
    print(f"  Done. Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
