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
# Dropout OFF. At 0.1 the model could not memorize even 64 grids at high
# mask ratios (weighted CE stuck ~1.0) yet crushed the same task at 0.0
# (0.03): corrupting 10% of activations makes the high-precision
# caption->content route unreliable, so training settles for the robust
# caption-agnostic texture solution in exactly the regime generation
# runs in (decode starts 100% masked). Overfitting is guarded by the
# masking itself — every sample is seen through a fresh random mask
# each epoch.
DROPOUT = 0.0
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

# CLIP model used to RE-RANK rendered ASCII grids (best-of-k). None =
# fall back to TEXT_ENCODER. Vanilla CLIP scores ASCII art at CHANCE
# level (ASCIIBench, arXiv 2512.04125 measured it — the representation, not the
# art, is the problem), so the stock re-ranker picks best-of-k nearly
# at random; that paper releases ASCII-aware fine-tuned CLIP weights —
# set their HF id here when fetched. The data engine's consistency
# filter is NOT affected (it scores the source *image*, where vanilla
# CLIP is fine) — this knob only changes rendered-grid scoring.
RANKER_MODEL = None

# Classifier-free guidance (inference default). Probe-calibrated: in
# this domain guidance is an object-count/literalness dial — ~1.5-2
# yields one clean subject, 3+ yields several (study guide 14.2). The
# old default of 3.0 was the multiple-crosses regime.
CFG_SCALE = 1.5
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
# Auto-calibrate the space weight per training stage to the dataset's
# actual space fraction f, as w = (1-f)/f (equal total gradient share
# for space and ink). 0.4 was hand-tuned on v1's ~75% space data and
# under-corrects badly on spacier data: at v3's ~90% space it left the
# space class with ~78% of the gradient — the attractor reborn.
SPACE_WEIGHT_AUTO = True

# Auxiliary perceptual loss through the differentiable glyph renderer:
# MSE between the rendered expectation over predicted glyphs and the
# rendered target, on masked cells. CE treats every wrong glyph as
# equally wrong; this term scores predictions by what they LOOK like,
# which is the actual quality criterion for ASCII art. 0 disables.
PERCEPTUAL_LOSS_WEIGHT = 0.1

# Classifier-free guidance schedule over the decode. "rise" starts at
# scale 1 on the empty canvas and grows to the full scale as cells
# commit (Muse used a linearly increasing schedule): layout forms
# without guidance amplifying the ink direction globally — the failure
# mode observed at scale >= 2 on v1 — while details still sharpen under
# full guidance. "constant" is classic CFG.
CFG_SCHEDULE = "rise"

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

# Self-context training (unrolled denoising, SUNDAE-style). With this
# probability per batch, a slice of the visible context is the model's
# OWN sampled predictions instead of ground truth, and the loss trains
# EVERY originally-masked cell — including the self-committed ones —
# against truth. Inference conditions every step on cells the model
# committed, but vanilla training only ever shows ground-truth context
# — so decode errors compound: several offset copies of a shape edge
# commit in parallel, each locally valid, and later steps extend them
# instead of suppressing them (nested-rectangle / interior-thatch
# "edge echo", observed at geometry_last even with gumbel 0 and 8
# revision steps). Keeping the committed cells in the loss is load-
# bearing: an ablation that scored only the still-masked cells made
# the echo WORSE (it taught the model that its own wrong commits are
# valid context to extend, and left revision-pass confidence at
# visible cells untrained). Costs one extra no-grad forward on
# affected batches. 0 disables.
SELF_CONTEXT_PROB = 0.5

# Stage 1: Geometry
# Geometry converges fast: observed train loss flat (0.034) from epoch 2
# on 200k samples. 6 epochs with a full cosine anneal beats paying for 15
# — the conditioning pathway keeps training through stages 2 and 3 anyway.
GEOMETRY_EPOCHS = 6
GEOMETRY_NUM_SAMPLES = 200_000
GEOMETRY_TRAIN_SAMPLES = 200_000

# Stage 2: Shading
# Raised 6 -> 30: masked models tolerate ~100+ epochs of data repetition
# (random masking is per-epoch augmentation; the data-constrained-scaling
# literature places best val loss for masked objectives at 10-50x the AR
# epoch count, with no overfitting inside budget), and rare subjects need
# on the order of ~1000 exposures to be stored at full fidelity — at 6
# epochs the sparse-subject tail is undertrained, which matches the
# observed weakness. Watch val loss per epoch; the anneal spans the full
# run, so stopping early wastes the schedule. (v1 note kept: a 100k
# truncation once silently dropped all samples past row 100k — None uses
# everything.)
SHADING_EPOCHS = 30
SHADING_NUM_SAMPLES = 100_000
SHADING_TRAIN_SAMPLES = None
SHADING_LR = 1e-4

# Stage 3: Human ASCII art fine-tune (optional — runs only if
# data/human_data.pt exists; see data/prepare_human_ascii.py)
# v1 ran 10 epochs and the human texture statistics bled into every
# prompt the model didn't understand; 3 epochs of style exposure keeps
# the benefit without the takeover.
HUMAN_EPOCHS = 3
HUMAN_LR = 5e-5
# Mix replayed shading samples into the human stage so fine-tuning on ~5k
# pieces doesn't erode the prompt conditioning learned on 100k captioned
# shading samples (0 disables)
HUMAN_REPLAY_SAMPLES = 20_000
# ... and replayed geometry too: v1 protected geometry only through the
# shading stage, and rectangles were gone by shading_last. Skills must be
# replayed in EVERY later stage, not just the next one.
HUMAN_GEOMETRY_REPLAY_SAMPLES = 10_000

# Replayed geometry mixed into the shading stage. v1 used 10k against a
# ~300k stage (3%) and geometry prompts came out BLANK at shading_last —
# diluted below survival. 60k (~20%) keeps primitives trainable.
SHADING_REPLAY_SAMPLES = 60_000

# Rare-caption upsampling: duplicate samples whose caption is rare so
# their subjects approach the ~1000-exposure storage threshold instead
# of drowning under the frequent ones. Each sample whose caption count
# is below the median unique-caption count is duplicated by
# ceil(median/count), capped at this factor. 1 disables. Duplication
# happens at index level before sharding, so every DDP rank sees the
# same expanded dataset; masking still differs per epoch, so duplicates
# are NOT identical training examples.
RARE_UPSAMPLE_MAX_FACTOR = 5

# Weight on the Token-Critic head's BCE loss (0 disables). The critic
# learns, per visible cell, "is this glyph correct?" — ground-truth
# visible cells are positives; the self-context corruption's revealed
# cells supply the negatives (model samples that missed the target) at
# zero extra cost. At decode it replaces generator confidence for
# commit ranking and revision re-masking (Token-Critic, arXiv
# 2209.04439: ImageNet FID 6.56 -> 4.69 over the same MaskGIT).
# The critic term is computed on EVERY batch (all-ones labels on plain
# batches) so the head's parameters join every DDP backward — see the
# conditionally-used-parameter crash, study guide 14.4.
CRITIC_LOSS_WEIGHT = 0.1

# Inference
UNMASK_STEPS = 10
TEMPERATURE = 1.0
# Cap on cells committed per unmasking step (None disables). The cosine
# schedule mass-commits the low-confidence tail in its final steps —
# precisely the ambiguous cells (e.g. every column an unseen shape edge
# could fall on) — sampling incompatible hypotheses in parallel: the
# edge-echo artifact. Capping resolves the tail sequentially, each
# commit becoming context for the next. Probe A/B on geometry_last:
# "a cross" went from 155-ink multi-stroke chaos to a perfect 28-ink
# cross; rectangle 300 -> 97 ink with one clean border. Costs extra
# forward passes (~masked_cells / cap total).
DECODE_MAX_COMMIT = 8
# ... but only in the decode's TAIL: the cap binds once the mask ratio
# falls to this threshold (1.0 = whole decode). Capping the head
# starved sparse tonal prompts into blank collapse — greedy space-
# cascade, with exploration noise and the rise schedule both expiring
# 6% into a ~480-step capped decode ("a dragon": 8 ink fully capped
# vs 865 uncapped, same checkpoint). Echo is born in the mass-
# committed ambiguous tail, not the head, so the head keeps the
# original explore-and-rise dynamics and the tail gets sequenced.
DECODE_CAP_BELOW = 0.35

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
