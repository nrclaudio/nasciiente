"""
Evaluation and visualization utilities.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE, DEVICE, BATCH_SIZE
from model.ascii_bert import ASCIIBert
from model.inference import generate
from training.dataset import ASCIIDataset
from training.masking import random_mask
from data.charset import grid_to_string, MASK_TOKEN
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_val(model, data, device, num_batches=50):
    """Compute validation loss and accuracy on a data tensor."""
    dataset = ASCIIDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    count = 0

    for masked_grid, target_grid, mask, ratio, label in loader:
        if count >= num_batches:
            break
        masked_grid = masked_grid.to(device)
        target_grid = target_grid.to(device)
        mask = mask.to(device)
        ratio = ratio.to(device)
        label = label.to(device)

        logits = model(masked_grid, class_label=label, mask_ratio=ratio)
        loss = model.compute_loss(logits, target_grid, mask)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        total_correct += (preds[mask] == target_grid[mask]).sum().item()
        total_masked += mask.sum().item()
        count += 1

    avg_loss = total_loss / max(count, 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


def generate_samples(model, device, n=5):
    """Generate n samples from fully masked grids."""
    print(f"\n=== Generating {n} samples ===\n")
    for i in range(n):
        _, final = generate(model, GRID_H, GRID_W,
                            num_steps=UNMASK_STEPS, temperature=TEMPERATURE,
                            device=device)
        print(f"--- Sample {i} ---")
        print(grid_to_string(final))
        print()


def inpainting_demo(model, data, device, n=3):
    """Mask a rectangular region in real samples and let the model fill it."""
    print(f"\n=== Inpainting Demo ({n} samples) ===\n")
    model.eval()

    for i in range(min(n, len(data))):
        original = data[i]  # [H, W]

        # Mask out a random rectangular region
        r0 = torch.randint(0, GRID_H // 2, (1,)).item()
        c0 = torch.randint(0, GRID_W // 2, (1,)).item()
        rh = torch.randint(GRID_H // 6, GRID_H // 3, (1,)).item()
        cw = torch.randint(GRID_W // 6, GRID_W // 3, (1,)).item()
        r1 = min(r0 + rh, GRID_H)
        c1 = min(c0 + cw, GRID_W)

        partial = original.clone()
        partial[r0:r1, c0:c1] = MASK_TOKEN

        _, filled = generate(model, GRID_H, GRID_W,
                             num_steps=UNMASK_STEPS, temperature=TEMPERATURE,
                             initial_grid=partial, device=device)

        print(f"--- Original {i} ---")
        print(grid_to_string(original))
        print(f"\n--- Masked (rows {r0}-{r1}, cols {c0}-{c1}) ---")
        print(grid_to_string(partial))
        print(f"\n--- Filled ---")
        print(grid_to_string(filled))
        print()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", help="Path to data .pt file (for inpainting demo)")
    args = parser.parse_args()

    device = DEVICE
    print(f"Device: {device}")

    model = ASCIIBert().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    # strict=False so pre-conditioning checkpoints load (their zero-init
    # conditioning is a no-op), but only conditioning params may be absent —
    # anything else is a genuinely mismatched checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if "conditioning" not in k] + list(unexpected)
    if bad:
        raise ValueError(f"Checkpoint {args.checkpoint} does not match the "
                         f"model (mismatched keys: {bad[:5]}...)")
    print("Model loaded.")

    # Generate from scratch
    generate_samples(model, device, n=5)

    # Validation + inpainting if data provided
    if args.data:
        data = torch.load(args.data, weights_only=True)
        if isinstance(data, dict):  # new label-carrying format
            data = data["data"]
        # dataset files store grids as uint8; embeddings need long indices
        data = data.long()
        val_loss, val_acc = evaluate_val(model, data, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.3f}")
        inpainting_demo(model, data, device, n=3)


if __name__ == "__main__":
    main()
