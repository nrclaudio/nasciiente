import torch

# Grid dimensions
GRID_H = 48
GRID_W = 80
GRID_SIZE = GRID_H * GRID_W  # 3840

# Character set
# Index 0 = [PAD], Index 1 = [MASK], Index 2+ = printable chars
VOCAB_SIZE = 98  # 96 printable ASCII + [PAD] + [MASK]

# Model architecture
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.1
MAX_ROWS = 96   # max supported grid height (for RoPE precomputation)
MAX_COLS = 160  # max supported grid width (allows 2x upscale of 48x80)

# Conditioning (Pack B). The model always accepts an optional class label
# and the current mask ratio; unlabeled data uses the null class. Class
# labels come from ImageNet-Sketch (1000 classes); index NUM_CLASSES is
# the null/unconditional token used for CFG and unlabeled data.
NUM_CLASSES = 1000
NULL_CLASS = NUM_CLASSES  # embedding index for "no class"

# Classifier-free guidance (inference default)
CFG_SCALE = 3.0
# Probability of dropping the class label to null during training (enables CFG)
CLASS_DROPOUT = 0.1

# Glyph-aware soft labels: blend one-hot targets with a visual-similarity
# distribution over glyphs, so confusing '/' with '|' costs less than
# confusing '/' with '@'. 0 disables (plain cross-entropy).
GLYPH_LABEL_SMOOTH = 0.1

# EMA of weights for evaluation/final checkpoint (0 disables)
EMA_DECAY = 0.999

# Mixed precision (bf16 autocast on CUDA; ignored on CPU)
USE_BF16 = True

# Training
BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 1  # no accumulation needed on A100
LEARNING_RATE = 3e-4
# Multiply stage LRs by the DDP world size (linear scaling rule): N GPUs
# means an N-times-larger effective batch, so N-times-fewer optimizer
# steps per epoch — larger steps compensate. Warmup keeps it stable.
SCALE_LR_WITH_GPUS = True
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
WARMUP_STEPS = 500
MASK_RATIO_MIN = 0.15
MASK_RATIO_MAX = 0.85

# Stage 1: Geometry
GEOMETRY_EPOCHS = 15
GEOMETRY_NUM_SAMPLES = 200_000
GEOMETRY_TRAIN_SAMPLES = 200_000

# Stage 2: Shading
SHADING_EPOCHS = 15
SHADING_NUM_SAMPLES = 100_000
SHADING_TRAIN_SAMPLES = 100_000
SHADING_LR = 1e-4

# Stage 3: Human ASCII art fine-tune (optional — runs only if
# data/human_data.pt exists; see data/prepare_human_ascii.py)
HUMAN_EPOCHS = 10
HUMAN_LR = 5e-5

# Inference
UNMASK_STEPS = 10
TEMPERATURE = 1.0

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
