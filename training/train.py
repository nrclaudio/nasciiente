"""
Two-stage curriculum training for ASCIIBert.

Stage 1: Geometry (learn structural primitives)
Stage 2: Shading  (fine-tune on density-mapped images)
"""

import math
import os
import sys
import time
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, DEVICE,
    GEOMETRY_EPOCHS, SHADING_EPOCHS, SHADING_LR, WARMUP_STEPS,
    GEOMETRY_TRAIN_SAMPLES, SHADING_TRAIN_SAMPLES,
    GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE,
    MASK_RATIO_MIN, MASK_RATIO_MAX,
)
from model.ascii_bert import ASCIIBert
from model.inference import generate
from training.dataset import ASCIIDataset
from data.charset import grid_to_string


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup followed by cosine decay to 0."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, scheduler, device, epoch, stage_name,
                total_epochs):
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    num_batches = 0
    total_batches = len(loader)
    log_interval = max(1, total_batches // 10)  # ~10 logs per epoch

    optimizer.zero_grad()
    t_epoch = time.time()

    for batch_idx, (masked_grid, target_grid, mask) in enumerate(loader):
        masked_grid = masked_grid.to(device, non_blocking=True)
        target_grid = target_grid.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        logits = model(masked_grid)
        loss = model.compute_loss(logits, target_grid, mask)
        (loss / GRAD_ACCUM_STEPS).backward()

        running_loss += loss.detach()
        num_batches += 1

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t_epoch
            avg = running_loss.item() / num_batches
            rate = (batch_idx + 1) / elapsed
            eta = (total_batches - batch_idx - 1) / rate
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  [{stage_name}] Epoch {epoch+1}/{total_epochs}  "
                  f"Batch {batch_idx+1:>5}/{total_batches}  "
                  f"Loss: {avg:.4f}  LR: {lr_now:.2e}  "
                  f"{rate:.1f} batch/s  ETA {eta:.0f}s",
                  flush=True)

    # Handle leftover gradients
    if num_batches > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return running_loss.item() / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for masked_grid, target_grid, mask in loader:
        masked_grid = masked_grid.to(device)
        target_grid = target_grid.to(device)
        mask = mask.to(device)

        logits = model(masked_grid)
        loss = model.compute_loss(logits, target_grid, mask)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)  # [B, H, W]
        total_correct += (preds[mask] == target_grid[mask]).sum().item()
        total_masked += mask.sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


def print_sample(model, device):
    """Generate and print a sample from a fully masked grid."""
    _, final = generate(model, GRID_H, GRID_W,
                        num_steps=UNMASK_STEPS, temperature=TEMPERATURE,
                        device=device)
    print("--- Generated Sample ---")
    print(grid_to_string(final))
    print()


def save_checkpoint(model, optimizer, epoch, stage, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
    }, path)


def train_stage(model, data_path, epochs, lr, stage_name, device, ckpt_dir,
                max_samples=None):
    print(f"\n{'='*60}")
    print(f"  Stage: {stage_name.upper()}")
    print(f"  Epochs: {epochs}, LR: {lr:.1e}, Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum")
    print(f"{'='*60}")

    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=True)
    if max_samples and max_samples < len(data):
        data = data[:max_samples]

    # 95/5 train/val split
    val_size = max(1, int(len(data) * 0.05))
    train_size = len(data) - val_size
    train_data, val_data = random_split(
        ASCIIDataset(data), [train_size, val_size]
    )
    print(f"Samples: {len(data):,} total -> {train_size:,} train / {val_size:,} val")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, persistent_workers=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, persistent_workers=True)
    print(f"Batches per epoch: {len(train_loader):,} train, {len(val_loader):,} val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * epochs
    warmup = min(WARMUP_STEPS, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    print(f"Scheduler: {warmup} warmup steps, {total_steps} total steps")

    t_stage = time.time()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, epoch, stage_name, epochs)
        val_loss, val_acc = validate(model, val_loader, device)

        elapsed = time.time() - t0
        stage_elapsed = time.time() - t_stage
        epochs_left = epochs - epoch - 1
        eta_stage = stage_elapsed / (epoch + 1) * epochs_left

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            marker = " *best*"

        print(f"\n[{stage_name}] Epoch {epoch+1}/{epochs} DONE — "
              f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
              f"Acc: {val_acc:.1%}{marker}")
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch time: {elapsed:.1f}s | "
              f"Stage ETA: {eta_stage/60:.1f}min | "
              f"LR: {lr_now:.2e}")

        # Print a generated sample each epoch
        print_sample(model, device)

        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{stage_name}_epoch{epoch+1}.pt")
        save_checkpoint(model, optimizer, epoch, stage_name, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    total_stage = time.time() - t_stage
    print(f"\n{'='*60}")
    print(f"  Stage {stage_name} complete in {total_stage/60:.1f}min "
          f"(best val loss: {best_val_loss:.4f})")
    print(f"{'='*60}")


def main():
    set_seed(42)
    t_total = time.time()

    device = DEVICE
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nConfig: batch={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, "
          f"eff_batch={BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Grid: {GRID_H}x{GRID_W} = {GRID_H * GRID_W} tokens/sample")
    print(f"Mask ratio: [{MASK_RATIO_MIN:.0%}, {MASK_RATIO_MAX:.0%}]")

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    model = ASCIIBert().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({trainable:,} trainable)")
    print(f"Gradient checkpointing: {model.transformer.gradient_checkpointing}")

    # Stage 1: Geometry
    geometry_path = os.path.join(data_dir, "geometry_data.pt")
    train_stage(model, geometry_path, GEOMETRY_EPOCHS, LEARNING_RATE,
                "geometry", device, ckpt_dir, max_samples=GEOMETRY_TRAIN_SAMPLES)

    # Stage 2: Shading
    shading_path = os.path.join(data_dir, "shading_data.pt")
    train_stage(model, shading_path, SHADING_EPOCHS, SHADING_LR,
                "shading", device, ckpt_dir, max_samples=SHADING_TRAIN_SAMPLES)

    # Save final model
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    torch.save(model.state_dict(), final_path)

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time/3600:.2f}h")
    print(f"  Final model: {final_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
