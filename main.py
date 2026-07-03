#!/usr/bin/env python3
"""
ASCII Art Transformer — single entry point for Vast.ai GPU instances.

Usage:
    python main.py                        # full pipeline (generate + train)
    python main.py --stage generate       # generate all datasets only
    python main.py --stage train          # train only (data must exist)
    python main.py --stage geometry       # generate geometry data only
    python main.py --stage shading        # generate shading data only
    python main.py --stage human          # prepare human ASCII art data only
    python main.py --source imagenette    # use Imagenette instead of sketches
    python main.py --num-shading 50000    # fewer shading samples
    python main.py --segment              # rembg background removal (photos)
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
        print(f"  VRAM:      {props.total_memory / 1e9:.1f} GB")
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


def run_generate_shading(source, num_samples, segment=False):
    """Generate image-derived shading data (Stage 2)."""
    from config import SHADING_NUM_SAMPLES
    from data.generate_shading import SOURCES, generate_dataset

    out_path = os.path.join(PROJECT_ROOT, "data", "shading_data.pt")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Shading data already exists ({size_mb:.0f} MB): {out_path}")
        print("  Delete it to regenerate. Skipping.\n")
        return

    num_samples = num_samples or SHADING_NUM_SAMPLES
    print(f"\n>>> Generating shading data ({source}: {SOURCES[source]})...\n")
    generate_dataset(source, num_samples, segment=segment, out_path=out_path)


def run_prepare_human(optional=True):
    """Download and normalize human ASCII art (optional Stage 3 data).

    With optional=True (the 'all'/'generate' paths) failures only print a
    warning so the rest of the pipeline still runs; with optional=False
    (explicit --stage human) failures propagate so the process exits nonzero.
    """
    out_path = os.path.join(PROJECT_ROOT, "data", "human_data.pt")
    if os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Human data already exists ({size_mb:.0f} MB): {out_path}")
        print("  Delete it to regenerate. Skipping.\n")
        return

    print("\n>>> Preparing human ASCII art data...\n")
    from data.prepare_human_ascii import main as human_main

    saved_argv = sys.argv
    sys.argv = [saved_argv[0]]
    try:
        human_main()
    # SystemExit too: prepare_human_ascii sys.exit(1)s when it finds no
    # usable art, and that must not abort the full pipeline before training
    except (Exception, SystemExit) as e:
        if not optional:
            raise
        reason = f"exit code {e.code}" if isinstance(e, SystemExit) else e
        print(f"WARNING: human data preparation failed ({reason}).")
        print("  Stage 3 will be skipped during training. "
              "Retry with: python data/prepare_human_ascii.py")
    finally:
        sys.argv = saved_argv
    print()


def run_train():
    """Run curriculum training, using all available GPUs."""
    import torch

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

    ngpu = torch.cuda.device_count()
    if ngpu > 1 and "RANK" not in os.environ:
        # Relaunch under torchrun so training/train.py runs one process
        # per GPU with DistributedDataParallel
        print(f"\n>>> {ngpu} GPUs detected — launching DDP via torchrun...\n")
        subprocess.run(
            [sys.executable, "-m", "torch.distributed.run", "--standalone",
             f"--nproc_per_node={ngpu}",
             os.path.join(PROJECT_ROOT, "training", "train.py")],
            check=True)
        return

    from training.train import main as train_main
    print("\n>>> Starting training...\n")
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
        choices=["all", "generate", "geometry", "shading", "human", "train"],
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--source", default="imagenet_sketch",
        choices=["imagenet_sketch", "imagenet", "imagenette", "caltech101",
                 "caltech256", "stl10"],
        help="Image source for shading data (default: imagenet_sketch)",
    )
    parser.add_argument(
        "--num-shading", type=int, default=None,
        help="Number of shading samples (default: from config)",
    )
    parser.add_argument(
        "--segment", action="store_true",
        help="Remove image backgrounds with rembg before ASCII conversion "
             "(requires: pip install rembg onnxruntime)",
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
        run_generate_shading(args.source, args.num_shading,
                             segment=args.segment)

    if args.stage in ("all", "generate", "human"):
        run_prepare_human(optional=args.stage != "human")

    if args.stage in ("all", "train"):
        run_train()

    elapsed = time.time() - t0
    print("=" * 64)
    print(f"  Done. Total time: {elapsed/3600:.2f}h ({elapsed:.0f}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
