"""
Curriculum training for ASCIIBert.

Stage 1: Geometry (learn structural primitives)
Stage 2: Shading  (fine-tune on density-mapped images)
Stage 3: Human ASCII art (optional fine-tune, if data/human_data.pt exists)

Single-GPU:  python training/train.py
Multi-GPU:   torchrun --standalone --nproc_per_node=NUM_GPUS training/train.py

Multi-GPU uses DistributedDataParallel (DDP): every GPU holds a full model
replica and processes a different shard of each batch; gradients are
averaged across replicas after every backward pass, so the effective batch
size is BATCH_SIZE * num_gpus. torchrun sets RANK/WORLD_SIZE/LOCAL_RANK in
the environment — when they are absent this script runs single-process.
"""

import math
import os
import sys
import time
import random

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    BATCH_SIZE, GRAD_ACCUM_STEPS, LEARNING_RATE, WEIGHT_DECAY, GRAD_CLIP, DEVICE,
    GEOMETRY_EPOCHS, SHADING_EPOCHS, SHADING_LR, WARMUP_STEPS,
    HUMAN_EPOCHS, HUMAN_LR, SCALE_LR_WITH_GPUS,
    GEOMETRY_TRAIN_SAMPLES, SHADING_TRAIN_SAMPLES,
    GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE,
    MASK_RATIO_MIN, MASK_RATIO_MAX,
    COND_DROPOUT, GLYPH_LABEL_SMOOTH, EMA_DECAY, USE_BF16, TEXT_ENCODER,
    CFG_SCALE, HUMAN_REPLAY_SAMPLES, SHADING_REPLAY_SAMPLES,
    HUMAN_GEOMETRY_REPLAY_SAMPLES, SPACE_LOSS_WEIGHT, SPACE_WEIGHT_AUTO,
)
from model.ascii_bert import ASCIIBert
from model.inference import generate
from training.dataset import ASCIIDataset
from data.charset import grid_to_string
from data.glyph_sim import build_soft_target_matrix


class EMA:
    """Exponential moving average of model weights.

    A slowly-tracked copy of the weights is a near-free way to stabilize
    sample quality — the EMA weights are used for validation, samples, and
    the saved checkpoints, while training continues on the raw weights.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: p.detach().clone()
                       for k, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[k].mul_(self.decay).add_(p.detach(),
                                                     alpha=1 - self.decay)

    def store_and_copy(self, model):
        """Swap live weights out for EMA weights (remember the live ones)."""
        self.backup = {}
        for k, p in model.named_parameters():
            if k in self.shadow:
                self.backup[k] = p.detach().clone()
                p.data.copy_(self.shadow[k])

    def restore(self, model):
        for k, p in model.named_parameters():
            if k in self.backup:
                p.data.copy_(self.backup[k])
        self.backup = {}


def _load_data_and_captions(path):
    """Load a dataset file.

    Current files are {'data', 'caption_ids', 'captions'} dicts: uint8
    grids, a per-sample index into the unique-caption list (-1 = no
    caption), and the captions themselves. Legacy formats — bare [N,H,W]
    tensors and the interim {'data','labels'} class-id dicts — load as
    uncaptioned (regenerate them to get text conditioning).

    Returns (data, caption_ids or None, captions or None).
    """
    obj = torch.load(path, weights_only=True)
    if isinstance(obj, dict):
        data = obj["data"]
        if "captions" in obj and "caption_ids" in obj:
            return data, obj["caption_ids"], list(obj["captions"])
        return data, None, None
    return obj, None, None


def _mix_replay(data, caption_ids, captions, replay_path, replay_samples):
    """Append a fixed random slice of another dataset (with its captions).

    Caption tables are concatenated and the replay slice's ids offset into
    the combined table. The selection uses a fixed generator so every DDP
    rank builds the identical mix.
    """
    r_data, r_ids, r_caps = _load_data_and_captions(replay_path)
    take = min(replay_samples, len(r_data))
    sel = torch.randperm(len(r_data),
                         generator=torch.Generator().manual_seed(7))[:take]

    captions = list(captions) if captions else []
    if caption_ids is None:
        caption_ids = torch.full((len(data),), -1, dtype=torch.long)
    if r_ids is None:
        r_sel_ids = torch.full((take,), -1, dtype=torch.long)
        r_caps = []
    else:
        r_sel_ids = r_ids[sel].long().clone()
        r_caps = list(r_caps) if r_caps else []
        # Offset replay ids into the merged caption table
        r_sel_ids[r_sel_ids >= 0] += len(captions)

    data = torch.cat([data.to(torch.uint8), r_data[sel].to(torch.uint8)])
    caption_ids = torch.cat([caption_ids.long(), r_sel_ids])
    captions = captions + r_caps
    return data, caption_ids, (captions if captions else None)


# Filled in by setup_distributed(); defaults describe a single process.
RANK = 0
WORLD_SIZE = 1
LOCAL_RANK = 0


def setup_distributed():
    """Initialize DDP when launched by torchrun; no-op otherwise.

    Returns (rank, world_size, local_rank).
    """
    global RANK, WORLD_SIZE, LOCAL_RANK
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(LOCAL_RANK)
    return RANK, WORLD_SIZE, LOCAL_RANK


def cleanup_distributed():
    if WORLD_SIZE > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


def unwrap(model):
    """Return the raw module whether or not it is wrapped in DDP."""
    return model.module if hasattr(model, "module") else model


def scale_lr(lr):
    """Linear LR scaling for DDP: effective batch grows with world size,
    so the learning rate grows with it (Goyal et al., 2017)."""
    if SCALE_LR_WITH_GPUS and WORLD_SIZE > 1:
        return lr * WORLD_SIZE
    return lr


def log(*args, **kwargs):
    """Print from the main process only, so N GPUs don't log N times."""
    if RANK == 0:
        print(*args, **kwargs)


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


def _autocast(device):
    """bf16 autocast on CUDA (no GradScaler needed), no-op elsewhere."""
    enabled = USE_BF16 and device.type == "cuda"
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                          enabled=enabled)


def train_epoch(model, loader, optimizer, scheduler, device, epoch, stage_name,
                total_epochs, sampler=None, soft_target=None, ema=None,
                space_weight=SPACE_LOSS_WEIGHT):
    if sampler is not None:
        # Reshuffle the shard assignment each epoch, or every rank would
        # see the same samples in the same order all run long
        sampler.set_epoch(epoch)
    model.train()
    running_loss = torch.tensor(0.0, device=device)
    num_batches = 0
    total_batches = len(loader)
    log_interval = max(1, total_batches // 10)  # ~10 logs per epoch

    optimizer.zero_grad()
    t_epoch = time.time()

    for batch_idx, (masked_grid, target_grid, mask, ratio, cond_tokens,
                    cond_mask, has_cond) in enumerate(loader):
        masked_grid = masked_grid.to(device, non_blocking=True)
        target_grid = target_grid.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        ratio = ratio.to(device, non_blocking=True)
        cond_tokens = cond_tokens.to(device, non_blocking=True)
        cond_mask = cond_mask.to(device, non_blocking=True)
        has_cond = has_cond.to(device, non_blocking=True)

        # Uncaptioned samples always use the learned null token; captioned
        # ones drop to null with prob COND_DROPOUT so the model learns
        # both distributions (classifier-free guidance).
        drop = ~has_cond
        if COND_DROPOUT > 0:
            drop = drop | (torch.rand(has_cond.shape, device=device)
                           < COND_DROPOUT)

        # Forward through the wrapper (DDP hooks gradient sync into
        # backward); compute_loss lives on the raw module
        with _autocast(device):
            logits = model(masked_grid, cond_tokens=cond_tokens,
                           cond_mask=cond_mask, cond_drop=drop,
                           mask_ratio=ratio)
            loss = unwrap(model).compute_loss(logits, target_grid, mask,
                                              soft_target_matrix=soft_target,
                                              space_weight=space_weight)
        (loss / GRAD_ACCUM_STEPS).backward()

        running_loss += loss.detach()
        num_batches += 1

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(unwrap(model))

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t_epoch
            avg = running_loss.item() / num_batches
            rate = (batch_idx + 1) / elapsed
            eta = (total_batches - batch_idx - 1) / rate
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  [{stage_name}] Epoch {epoch+1}/{total_epochs}  "
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
        if ema is not None:
            ema.update(unwrap(model))

    return running_loss.item() / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, device, soft_target=None,
             space_weight=SPACE_LOSS_WEIGHT):
    # Use the raw module: DDP's forward broadcasts buffers (a collective
    # op), which would deadlock if ranks ever ran different batch counts.
    # Every rank validates the identical full val split instead — the 5%
    # split is cheap and all ranks arrive at the same metrics.
    model = unwrap(model)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0

    for (masked_grid, target_grid, mask, ratio, cond_tokens, cond_mask,
         has_cond) in loader:
        masked_grid = masked_grid.to(device)
        target_grid = target_grid.to(device)
        mask = mask.to(device)
        ratio = ratio.to(device)
        cond_tokens = cond_tokens.to(device)
        cond_mask = cond_mask.to(device)
        has_cond = has_cond.to(device)

        with _autocast(device):
            logits = model(masked_grid, cond_tokens=cond_tokens,
                           cond_mask=cond_mask, cond_drop=~has_cond,
                           mask_ratio=ratio)
            loss = model.compute_loss(logits, target_grid, mask,
                                      soft_target_matrix=soft_target,
                                      space_weight=space_weight)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)  # [B, H, W]
        total_correct += (preds[mask] == target_grid[mask]).sum().item()
        total_masked += mask.sum().item()

    avg_loss = total_loss / max(len(loader), 1)
    accuracy = total_correct / max(total_masked, 1)
    return avg_loss, accuracy


def print_sample(model, device, save_path=None, probe=None):
    """Generate and print per-epoch samples (rank 0 only).

    Always generates unconditionally; with probe=(prompt_text, cond_tokens,
    cond_mask) it also generates a prompted sample with guidance. The
    unconditional mode only sees COND_DROPOUT of the training signal on
    captioned data (and sparse grids collapse it toward all-blank early
    on), so the prompted sample is the meaningful progress signal for
    conditioned stages.

    With save_path, the samples are also written to disk — one file per
    epoch under checkpoints/samples/, which the Streamlit app renders as
    a training-progress gallery.
    """
    if RANK != 0:
        return
    _, final = generate(unwrap(model), GRID_H, GRID_W,
                        num_steps=UNMASK_STEPS, temperature=TEMPERATURE,
                        device=device)
    sections = [("unconditional", final)]
    if probe is not None:
        text, toks, msk = probe
        _, prompted = generate(unwrap(model), GRID_H, GRID_W,
                               num_steps=UNMASK_STEPS,
                               temperature=TEMPERATURE, device=device,
                               cond_tokens=toks, cond_mask=msk,
                               guidance_scale=CFG_SCALE)
        sections.append((f'prompt: "{text}" (guidance {CFG_SCALE})',
                         prompted))
    for title, grid in sections:
        print(f"--- Generated Sample ({title}) ---")
        print(grid_to_string(grid))
        print()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write("\n\n".join(f"=== {title} ===\n{grid_to_string(grid)}"
                                for title, grid in sections))


def save_checkpoint(model, optimizer, epoch, stage, path, ema=None):
    if RANK != 0:
        return
    ckpt = {
        # unwrap so checkpoints have identical keys with and without DDP.
        # model_state_dict holds the live training weights (they match
        # optimizer_state_dict, so a resume continues where it left off);
        # the EMA weights used for eval/deployment ride along separately.
        "model_state_dict": unwrap(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "stage": stage,
    }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.shadow
    torch.save(ckpt, path)


def auto_space_weight(data):
    """Loss weight for space cells giving space and ink equal total
    gradient share: w = (1-f)/f for space fraction f. Reproduces the
    hand-tuned 0.4 on v1-era data (~75% space) and correctly shrinks on
    spacier corpora, where a fixed 0.4 leaves the blank-collapse prior
    in charge."""
    from model.ascii_bert import SPACE_IDX
    f = float((data == SPACE_IDX).float().mean())
    if f <= 0 or f >= 1:
        return 1.0
    return min(1.0, max(0.05, (1 - f) / f))


def train_stage(model, data_path, epochs, lr, stage_name, device, ckpt_dir,
                max_samples=None, ema=None, soft_target=None,
                replay_path=None, replay_samples=0):
    base_lr, lr = lr, scale_lr(lr)
    log(f"\n{'='*60}")
    log(f"  Stage: {stage_name.upper()}")
    log(f"  Epochs: {epochs}, LR: {lr:.1e}"
        + (f" (base {base_lr:.1e} x {WORLD_SIZE} GPUs)" if lr != base_lr else "")
        + f", Batch: {BATCH_SIZE} x {GRAD_ACCUM_STEPS} accum x {WORLD_SIZE} GPU(s) "
          f"= {BATCH_SIZE * GRAD_ACCUM_STEPS * WORLD_SIZE} effective")
    log(f"{'='*60}")

    log(f"Loading data from {data_path}...")
    data, caption_ids, captions = _load_data_and_captions(data_path)
    if max_samples and max_samples < len(data):
        data = data[:max_samples]
        if caption_ids is not None:
            caption_ids = caption_ids[:max_samples]
    # Replay accepts one source or several [(path, n), ...] — skills must
    # be replayed in EVERY later stage, not just the next one (geometry
    # was gone by shading_last in v1 with single-hop replay)
    if replay_path:
        pairs = (list(zip(replay_path, replay_samples))
                 if isinstance(replay_path, (list, tuple))
                 else [(replay_path, replay_samples)])
        for r_path, r_n in pairs:
            if r_n > 0 and os.path.exists(r_path):
                n_before = len(data)
                data, caption_ids, captions = _mix_replay(
                    data, caption_ids, captions, r_path, r_n)
                log(f"  Replay: +{len(data) - n_before:,} samples from "
                    f"{os.path.basename(r_path)} "
                    f"(guards against forgetting)")
    space_weight = (auto_space_weight(data) if SPACE_WEIGHT_AUTO
                    else SPACE_LOSS_WEIGHT)
    log(f"  Space weight: {space_weight:.3f} "
        f"({'auto — equal gradient share' if SPACE_WEIGHT_AUTO else 'fixed'})")
    caption_tokens = caption_masks = None
    if captions is not None:
        # Embed the unique-caption vocabulary once up front; the training
        # loop then only indexes into this table (the encoder never runs
        # inside the loop)
        log(f"  Captions: {len(captions):,} unique — embedding with "
            f"{TEXT_ENCODER}...")
        from data.text_embed import embed_captions
        caption_tokens, caption_masks = embed_captions(captions)
        log(f"  Caption tokens: {tuple(caption_tokens.shape)} "
            f"({caption_tokens.nbytes / 1e6:.0f} MB)")
    else:
        log("  Captions: none (unconditional stage)")

    # 95/5 train/val split — fixed generator so every rank builds the
    # identical split regardless of its own RNG state
    val_size = max(1, int(len(data) * 0.05))
    train_size = len(data) - val_size
    train_data, val_data = random_split(
        ASCIIDataset(data, caption_ids, caption_tokens, caption_masks),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"Samples: {len(data):,} total -> {train_size:,} train / {val_size:,} val")

    # Under DDP each rank gets a disjoint 1/WORLD_SIZE shard per epoch
    train_sampler = (DistributedSampler(train_data, shuffle=True)
                     if WORLD_SIZE > 1 else None)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler,
                              num_workers=4, persistent_workers=True,
                              prefetch_factor=4)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, persistent_workers=True)
    log(f"Batches per epoch: {len(train_loader):,} train, {len(val_loader):,} val")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * epochs
    warmup = min(WARMUP_STEPS, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    log(f"Scheduler: {warmup} warmup steps, {total_steps} total steps")

    t_stage = time.time()
    best_val_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                 device, epoch, stage_name, epochs,
                                 sampler=train_sampler, soft_target=soft_target,
                                 ema=ema, space_weight=space_weight)

        # Evaluate and sample under EMA weights (swap in, then restore the
        # live training weights before checkpointing so the saved weights
        # match the saved optimizer state)
        if ema is not None:
            ema.store_and_copy(unwrap(model))
        val_loss, val_acc = validate(model, val_loader, device, soft_target,
                                     space_weight=space_weight)

        elapsed = time.time() - t0
        stage_elapsed = time.time() - t_stage
        epochs_left = epochs - epoch - 1
        eta_stage = stage_elapsed / (epoch + 1) * epochs_left

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss

        log(f"\n[{stage_name}] Epoch {epoch+1}/{epochs} DONE — "
            f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, "
            f"Acc: {val_acc:.1%}{' *best*' if improved else ''}")
        lr_now = optimizer.param_groups[0]["lr"]
        log(f"  Epoch time: {elapsed:.1f}s | "
            f"Stage ETA: {eta_stage/60:.1f}min | "
            f"LR: {lr_now:.2e}")

        # Print generated samples each epoch, and archive them so the app
        # can show how samples evolve over the curriculum. For captioned
        # stages, probe with the stage's first caption so the prompted
        # mode (the one 90% of training optimizes) is visible too.
        sample_path = os.path.join(ckpt_dir, "samples",
                                   f"{stage_name}_epoch{epoch+1:03d}.txt")
        probe = ((captions[0], caption_tokens[0], caption_masks[0])
                 if caption_tokens is not None else None)
        print_sample(model, device, save_path=sample_path, probe=probe)

        if ema is not None:
            ema.restore(unwrap(model))

        # Rolling checkpoints: 'last' every epoch (crash recovery) and
        # 'best' on val improvement. Per-epoch files would pile up to
        # ~12 GB over a full curriculum for no benefit.
        last_path = os.path.join(ckpt_dir, f"{stage_name}_last.pt")
        save_checkpoint(model, optimizer, epoch, stage_name, last_path,
                        ema=ema)
        if improved:
            save_checkpoint(model, optimizer, epoch, stage_name,
                            os.path.join(ckpt_dir, f"{stage_name}_best.pt"),
                            ema=ema)
        log(f"  Checkpoint saved: {last_path}"
            + (" (+ best)" if improved else ""))

    total_stage = time.time() - t_stage
    log(f"\n{'='*60}")
    log(f"  Stage {stage_name} complete in {total_stage/60:.1f}min "
        f"(best val loss: {best_val_loss:.4f})")
    log(f"{'='*60}")


def parse_args():
    """CLI for partial runs: initialize from a checkpoint and/or run a
    subset of stages, so training-recipe changes don't cost a full
    curriculum rerun (e.g. --init-from checkpoints/geometry_best.pt
    --stages shading,human). parse_known_args so main.py can call
    train_main() in-process with its own argv."""
    import argparse
    # allow_abbrev=False: main.py calls train_main() in-process with its own
    # --stage flag, which prefix-matching would otherwise swallow as --stages
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--init-from", default=None,
                        help="checkpoint to initialize model weights from")
    parser.add_argument("--stages", default="geometry,shading,human",
                        help="comma-separated stages to run")
    parser.add_argument("--shading-data", default=None,
                        help="alternative dataset file for the shading "
                             "stage (e.g. data/synthetic_data.pt from the "
                             "data engine)")
    parser.add_argument("--model-size", default="base",
                        choices=["base", "large"],
                        help="base = 34.5M (original), large = ~113M "
                             "(768x12 — for the high-entropy tonal "
                             "dialect). probe/server auto-detect the "
                             "size from checkpoint shapes.")
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    rank, world_size, local_rank = setup_distributed()
    # Different seed per rank for masking/augmentation randomness; the
    # train/val split uses its own fixed generator so it stays identical
    set_seed(42 + rank)
    t_total = time.time()

    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}"
                              if torch.cuda.is_available() else "cpu")
        log(f"DDP: {world_size} processes "
            f"({'NCCL' if torch.cuda.is_available() else 'gloo'} backend)")
    else:
        device = DEVICE
    log(f"Device: {device}")
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(local_rank)}")
        log(f"GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")

    log(f"\nConfig: batch={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, "
        f"world={world_size}, "
        f"eff_batch={BATCH_SIZE * GRAD_ACCUM_STEPS * world_size}")
    log(f"Grid: {GRID_H}x{GRID_W} = {GRID_H * GRID_W} tokens/sample")
    log(f"Mask ratio: [{MASK_RATIO_MIN:.0%}, {MASK_RATIO_MAX:.0%}]")

    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    from model.ascii_bert import MODEL_SIZES
    model = ASCIIBert(**MODEL_SIZES[args.model_size]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Model size: {args.model_size} "
        f"({MODEL_SIZES[args.model_size]})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"\nModel: {total_params:,} params ({trainable:,} trainable)")
    log(f"Gradient checkpointing: {model.transformer.gradient_checkpointing}")

    if args.init_from:
        from model.ascii_bert import load_compatible_state
        ckpt = torch.load(args.init_from, map_location=device,
                          weights_only=True)
        state = (ckpt["model_state_dict"]
                 if isinstance(ckpt, dict) and "model_state_dict" in ckpt
                 else ckpt)
        load_compatible_state(model, state)
        log(f"Initialized weights from {args.init_from}")

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # DDP broadcasts rank 0's weights at construction, so all
        # replicas start identical even with per-rank seeds
        model = DDP(model, device_ids=[local_rank]
                    if torch.cuda.is_available() else None)

    # EMA weights (stabilize samples) and glyph-aware soft targets
    ema = EMA(unwrap(model), EMA_DECAY) if EMA_DECAY > 0 else None
    soft_target = None
    if GLYPH_LABEL_SMOOTH > 0:
        soft_target = build_soft_target_matrix(GLYPH_LABEL_SMOOTH).to(device)
        is_identity = torch.allclose(
            soft_target, torch.eye(soft_target.shape[0], device=device))
        log(f"Glyph soft targets: "
            f"{'identity (no font — plain CE)' if is_identity else f'alpha={GLYPH_LABEL_SMOOTH}'}")
    log(f"EMA: {'decay ' + str(EMA_DECAY) if ema else 'off'} | "
        f"bf16: {USE_BF16 and device.type == 'cuda'} | "
        f"caption dropout: {COND_DROPOUT}")

    # Stage 1: Geometry
    geometry_path = os.path.join(data_dir, "geometry_data.pt")
    if "geometry" in stages:
        train_stage(model, geometry_path, GEOMETRY_EPOCHS, LEARNING_RATE,
                    "geometry", device, ckpt_dir,
                    max_samples=GEOMETRY_TRAIN_SAMPLES,
                    ema=ema, soft_target=soft_target)

    # Stage 2: Shading, with a slice of replayed geometry so stage-1
    # skills stay alive (and get re-trained under the current objective
    # when resuming from an older stage-1 checkpoint)
    shading_path = args.shading_data or os.path.join(data_dir,
                                                     "shading_data.pt")
    if "shading" in stages:
        train_stage(model, shading_path, SHADING_EPOCHS, SHADING_LR,
                    "shading", device, ckpt_dir,
                    max_samples=SHADING_TRAIN_SAMPLES,
                    ema=ema, soft_target=soft_target,
                    replay_path=geometry_path,
                    replay_samples=SHADING_REPLAY_SAMPLES)

    # Stage 3 (optional): fine-tune on human-made ASCII art, with slices
    # of replayed shading AND geometry so neither prompt conditioning
    # nor primitives erode
    human_path = os.path.join(data_dir, "human_data.pt")
    if "human" in stages and os.path.exists(human_path):
        train_stage(model, human_path, HUMAN_EPOCHS, HUMAN_LR,
                    "human", device, ckpt_dir, ema=ema,
                    soft_target=soft_target,
                    replay_path=[shading_path, geometry_path],
                    replay_samples=[HUMAN_REPLAY_SAMPLES,
                                    HUMAN_GEOMETRY_REPLAY_SAMPLES])
    elif "human" in stages:
        log("\nNo human_data.pt found — skipping stage 3 "
            "(run data/prepare_human_ascii.py to enable it)")

    # Save final model under EMA weights
    final_path = os.path.join(ckpt_dir, "final_model.pt")
    if RANK == 0:
        if ema is not None:
            ema.store_and_copy(unwrap(model))
        torch.save(unwrap(model).state_dict(), final_path)
        if ema is not None:
            ema.restore(unwrap(model))

    total_time = time.time() - t_total
    log(f"\n{'='*60}")
    log(f"  Training complete in {total_time/3600:.2f}h")
    log(f"  Final model: {final_path}")
    log(f"{'='*60}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
