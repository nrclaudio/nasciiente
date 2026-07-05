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

# Text conditioning. Prompts (dataset captions, class names, user text) are
# embedded with a frozen text encoder; the per-token hidden states feed
# cross-attention in every transformer block (compositional control) and
# their masked mean feeds a global additive vector. The model also always
# receives the current mask ratio. Samples without a caption train against
# a learned null token, which doubles as the unconditional branch for
# classifier-free guidance.
TEXT_ENCODER = "openai/clip-vit-base-patch32"  # frozen; only used to embed text
TEXT_EMB_DIM = 512                             # CLIP text hidden dim
TEXT_COND_TOKENS = 24  # max caption tokens kept for cross-attention

# Classifier-free guidance (inference default)
CFG_SCALE = 3.0
# Probability of dropping the caption to null during training (enables CFG)
COND_DROPOUT = 0.1

# Glyph-aware soft labels: blend one-hot targets with a visual-similarity
# distribution over glyphs, so confusing '/' with '|' costs less than
# confusing '/' with '@'. 0 disables (plain cross-entropy).
GLYPH_LABEL_SMOOTH = 0.1

# Down-weight space cells in the loss. ~80% of target cells are space, so
# plain CE makes "space" the low-risk answer everywhere — the exact prior
# behind blank-collapse in free generation. 1.0 disables.
SPACE_LOSS_WEIGHT = 0.4

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
# Up to fully masked: inference always STARTS from a 100%-masked grid, so
# training must cover that state or the first generation steps are
# out-of-distribution (which shows up as blank-grid collapse)
MASK_RATIO_MAX = 1.0

# Stage 1: Geometry
# Geometry converges fast: observed train loss flat (0.034) from epoch 2
# on 200k samples. 6 epochs with a full cosine anneal beats paying for 15
# — the conditioning pathway keeps training through stages 2 and 3 anyway.
GEOMETRY_EPOCHS = 6
GEOMETRY_NUM_SAMPLES = 200_000
GEOMETRY_TRAIN_SAMPLES = 200_000

# Stage 2: Shading
# 6 epochs: the shading stage now trains on ~310k samples (synthetic +
# sketch + geometry replay) — 6 epochs of that sees more unique content
# than 15 epochs of the old 100k set, at 40% of the cost
SHADING_EPOCHS = 6
SHADING_NUM_SAMPLES = 100_000
SHADING_TRAIN_SAMPLES = 100_000
SHADING_LR = 1e-4

# Stage 3: Human ASCII art fine-tune (optional — runs only if
# data/human_data.pt exists; see data/prepare_human_ascii.py)
HUMAN_EPOCHS = 10
HUMAN_LR = 5e-5
# Mix replayed shading samples into the human stage so fine-tuning on ~5k
# pieces doesn't erode the prompt conditioning learned on 100k captioned
# shading samples (0 disables)
HUMAN_REPLAY_SAMPLES = 20_000

# Likewise mix replayed geometry samples into the shading stage: keeps
# geometry-prompt skills alive and — when resuming from a stage-1
# checkpoint trained under an older objective — re-trains those samples
# under the current loss (0 disables)
SHADING_REPLAY_SAMPLES = 10_000

# Inference
UNMASK_STEPS = 10
TEMPERATURE = 1.0

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
