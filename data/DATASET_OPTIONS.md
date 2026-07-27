# Image dataset options for the shading stage

Comparison of candidate datasets to replace STL-10 for the shading
(stage 2) training data.

## Requirements

- Clear subjects with strong edges (ASCII art needs recognizable shapes)
- Good tonal range / contrast for shading
- Resolution ≥ 96×96 (upscaled to 576×640 for sub-cell matching)
- Enough images for 100k samples (with or without augmentation)
- Subject diversity (not just one category)

## Current: STL-10

| Property | Value |
|----------|-------|
| Resolution | 96×96 |
| Total images | ~113k (100k unlabeled + 13k labeled) |
| Classes | 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck) |
| Download | ~2.6 GB via `torchvision.datasets.STL10` |
| Auth | None |

**Problems**: cluttered backgrounds, small or off-center subjects, limited
tonal range in many images. 96×96 means a 20× upscale to the sub-cell
target, so most of it is interpolated.

---

## Option 1: ImageNet (ILSVRC 2012), recommended

| Property | Value |
|----------|-------|
| Resolution | Variable, avg ~469×387 |
| Total images | ~1.28M training |
| Classes | 1000 |
| Download | ~150 GB (full) or **stream via HuggingFace** |
| Auth | HuggingFace token + dataset agreement |

**Pros**: best diversity (1000 classes), highest resolution, huge pool to
sample from (only 100k of 1.28M needed, so high-contrast images can be
cherry-picked). Variable resolution means many images are 400+ pixels on a
side.

**Cons**: requires HuggingFace auth. Some classes are noisy, though that is
easy to skip with 1000 to choose from.

```python
# Streaming, no 150GB download needed
from datasets import load_dataset
ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True)
```

## Option 2: Imagenette

| Property | Value |
|----------|-------|
| Resolution | 320×320 (full) or 160×160 |
| Total images | ~13k |
| Classes | 10 (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute) |
| Download | ~1.5 GB (320px) |
| Auth | None |

**Pros**: clean, easily classifiable images (that's the point of "easy
ImageNet"). 320×320 is 3× STL resolution. No auth needed. Very easy to set
up.

**Cons**: only ~13k images, so hitting 100k samples needs heavy cycling and
augmentation. Only 10 classes, though they are well-chosen for visual
clarity.

```python
# Via HuggingFace
from datasets import load_dataset
ds = load_dataset("frgfm/imagenette", "320px", split="train")

# Or direct download
# https://github.com/fastai/imagenette
```

## Option 3: Caltech-101

| Property | Value |
|----------|-------|
| Resolution | Variable, ~200-300px typical |
| Total images | ~9,146 |
| Classes | 101 (animals, vehicles, objects, faces, etc.) |
| Download | ~137 MB via `torchvision.datasets.Caltech101` |
| Auth | None |

**Pros**: very clean, with objects well-centered on simple or white
backgrounds. Excellent edge definition. Good class diversity (101
categories). Tiny download.

**Cons**: small dataset (~9k), so it would need significant cycling and
augmentation. Variable image sizes.

```python
import torchvision
ds = torchvision.datasets.Caltech101(root="./data/caltech101", download=True)
```

## Option 4: Caltech-256

| Property | Value |
|----------|-------|
| Resolution | Variable, ~200-300px typical |
| Total images | ~30,607 |
| Classes | 257 (256 object categories + clutter) |
| Download | ~1.2 GB via `torchvision.datasets.Caltech256` |
| Auth | None |

**Pros**: same clean quality as Caltech-101 but 3× more images and 2.5×
more classes. Better diversity.

**Cons**: still only ~30k images against a 100k target. Variable sizes.

```python
import torchvision
ds = torchvision.datasets.Caltech256(root="./data/caltech256", download=True)
```

---

## Comparison matrix

| Dataset | Resolution | Images | Classes | Download | Auth | Quality |
|---------|-----------|--------|---------|----------|------|---------|
| STL-10 (current) | 96×96 | 113k | 10 | 2.6 GB | None | Noisy |
| **ImageNet** | ~469×387 | 1.28M | 1000 | Stream | HF token | High |
| **Imagenette** | 320×320 | 13k | 10 | 1.5 GB | None | High |
| **Caltech-101** | ~200-300 | 9k | 101 | 137 MB | None | Very high |
| **Caltech-256** | ~200-300 | 30k | 257 | 1.2 GB | None | Very high |

## Recommended approach

A multi-dataset mix, for the best coverage:

| Source | Sample count | Why |
|--------|-------------|-----|
| ImageNet (streamed) | 60k | Diversity + resolution backbone |
| Caltech-256 | 25k | Clean objects, strong edges (cycle ~1×) |
| Imagenette | 15k | High-res, clean subjects (cycle ~1×) |
| **Total** | **100k** | |

That covers 1000+ categories at consistently high resolution, with the
Caltech and Imagenette images raising the average quality.

**Fallback** (no HuggingFace auth): Caltech-256 (30k, cycled 2×) +
Imagenette (13k, cycled ~3×) + augmentation = 100k. Lower diversity but
zero auth friction.
