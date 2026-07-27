# nASCIIente — Study Guide

**Generating ASCII art from text: the complete story of a 34.5M-parameter model, from first principles.**

> This document explains, from the ground up, a system that turns a text prompt ("a goat") into a 48×80 grid of ASCII characters depicting that subject. It assumes no prior machine-learning knowledge: every concept — embeddings, attention, masked training, iterative decoding, classifier-free guidance — is introduced before it is used, with the actual formulas implemented in the code.
>
> The system has three parts: a **data engine** that manufactures a captioned training set (because none exists in the world), a **bidirectional transformer** of 34.5 million parameters trained from scratch to predict masked grid cells, and a **MaskGIT-style decoder** that generates a full picture in about ten parallel steps. Six documented failures shaped the v1 design and are analyzed in detail, because they are where the learning lives; the appendices carry the v2/v3 revisions and their own failures — including two of the deepest in the project (dropout silently vetoing prompt following, and exposure bias with a fix that backfired — Appendix C).

| | |
|---|---|
| Parameters | 34.5M, trained from scratch |
| Canvas | 48×80 = 3,840 cells |
| Vocabulary | 98 tokens (96 printable ASCII + `[PAD]` + `[MASK]`) |
| Architecture | 8 layers · 512 dim · 8 heads · 2D RoPE · cross-attention |
| Decoding | ~10 parallel unmasking steps |

**Contents**

- Part I — Foundations: §1 The task · §2 Neural networks in ten minutes · §3 Embeddings
- Part II — The model: §4 Attention · §5 Positions and 2D RoPE · §6 CLIP and cross-attention
- Part III — Training: §7 Masked prediction · §8 Optimization · §9 Curriculum
- Part IV — Generation: §10 MaskGIT decoding · §11 Classifier-free guidance · §12 Best-of-k
- Part V — Data: §13 The synthetic data engine
- Part VI — Lessons: §14 Six failures · §15 Decisions defended · §16 Glossary · §17 Self-test
- Appendix A: the v2 revision (training fixes)
- Appendix B: the v3 data engine (FLUX, dialects, composites)
- Appendix C: the v3 training run (dropout off, auto space weight, exposure bias & self-context)

---

## Part I — Foundations

### §1 The task, stated precisely

An ASCII picture is a rectangular arrangement of typed characters. Fix the canvas at $H = 48$ rows by $W = 80$ columns — 3,840 cells. Each cell holds exactly one symbol from a fixed **vocabulary** $V$: the 96 printable ASCII characters (space through `~`) plus two special bookkeeping symbols, `[PAD]` and `[MASK]`, for 98 symbols total. A picture is therefore a point in a finite (but astronomically large) space:

$$x \in V^{H \times W}, \qquad |V| = 98, \qquad \text{so there are } 98^{3840} \text{ possible grids} \tag{1.1}$$

"Generating ASCII art from a prompt" means: given a text caption $c$ (like "a goat"), draw a sample from a probability distribution over grids that puts high probability on grids humans would recognize as depicting $c$:

$$x \sim p_\theta(x \mid c) \tag{1.2}$$

Here $\theta$ stands for the model's parameters — the 34.5 million numbers we will learn from data. Everything in this document serves two questions: how to *represent* $p_\theta(x|c)$ with a neural network, and how to *sample* from it efficiently.

Note what this framing rejects: we are *not* generating an image of pixels and converting it to text afterward. The grid of discrete characters **is** the object being modeled. This makes the problem feel much more like language modeling (discrete symbols, exact positions) than like image generation (continuous intensities) — and that observation drives the entire design.

> **Why it matters.** $98^{3840}$ is incomprehensibly larger than the number of atoms in the universe. No table of good pictures can exist; the model must learn *rules and regularities* — strokes, symmetry, how a `/` continues into a `|` — that let it construct never-seen grids that still look right. That is what "learning a distribution" means in practice.

### §2 Neural networks in ten minutes

A neural network is a function with adjustable knobs. It takes numbers in, applies a sequence of simple operations — multiply by a matrix, add a vector, apply a fixed nonlinear function — and produces numbers out. The entries of the matrices and vectors are the **parameters** $\theta$. A single layer looks like:

$$h = \sigma(Wx + b) \tag{2.1}$$

where $W$ is a matrix of weights, $b$ a vector of biases, and $\sigma$ a nonlinearity applied element-wise (this model uses GELU, a smooth ramp). Stacking layers gives the network *depth*; the nonlinearities are what let depth express complicated functions rather than collapsing into one big matrix multiply.

**Logits and softmax: how a network expresses beliefs.** Our network's final output, for each of the 3,840 cells, is a vector of 98 raw scores called **logits** — one score per vocabulary symbol, higher meaning "more plausible here." Logits become a probability distribution through the **softmax**:

$$p(v) = \frac{\exp(z_v)}{\sum_{u \in V} \exp(z_u)} \tag{2.2}$$

Exponentiating makes every value positive; dividing by the sum makes them add to 1. Softmax preserves order (the highest logit gets the highest probability) and responds smoothly to changes in the logits, which is essential for learning.

**Temperature** $\tau$ rescales logits before the softmax and controls how adventurous sampling is:

$$p_\tau(v) = \operatorname{softmax}(z / \tau)_v \tag{2.3}$$

As $\tau \to 0$ the distribution sharpens toward pure argmax (always pick the top choice — safe, repetitive); $\tau > 1$ flattens it (more variety, more mistakes). This model defaults to $\tau = 1$.

**Loss and gradient descent: how the knobs get set.** Training needs a single number measuring how wrong the network currently is — the **loss** $L(\theta)$ — and a way to nudge every parameter to reduce it. The nudge direction comes from calculus: the **gradient** $\nabla_\theta L$ points uphill, so we step downhill:

$$\theta \leftarrow \theta - \eta \cdot \nabla_\theta L(\theta) \tag{2.4}$$

$\eta$ is the **learning rate** (here $3\times10^{-4}$ at the start, smaller in later stages). Computing the gradient through every layer is **backpropagation** — the chain rule, applied automatically by PyTorch. Because the true loss over the whole dataset is too expensive per step, we estimate it on a random **mini-batch** (64 grids) each step: stochastic gradient descent. In practice we use **Adam**, a refinement giving each parameter its own adaptive step size (§8).

> **Beginner's summary.** A model is a differentiable function with millions of knobs. Show it examples, measure wrongness with a loss, use gradients to turn every knob a tiny bit in the direction that reduces wrongness, repeat millions of times. Everything else is about *which function*, *which loss*, and *which examples*.

### §3 From characters to vectors: embeddings

Networks compute with continuous numbers, but our symbols are discrete. The bridge is the **embedding table**: a learned matrix $E \in \mathbb{R}^{98 \times 512}$ holding one 512-dimensional vector per vocabulary symbol. "Embedding a character" is just looking up its row:

$$\operatorname{embed}(v) = E_v \in \mathbb{R}^{512} \tag{3.1}$$

These vectors start random and are learned like any other parameter. During training the network discovers useful geometry on its own: characters that play similar visual roles (say `/` and `\`, or the heavy "ink" characters `@`, `#`, `8`) drift toward each other, because similar vectors let the rest of the network treat them similarly. Meaning becomes *direction in space*.

The number 512 is the model's **width** ($d_\text{model}$): every cell is represented by a 512-dimensional vector throughout the network. After embedding, the grid is a 48×80×512 tensor — each cell carries a 512-number "working memory" about what it is and, as computation proceeds, what is around it.

A note on scale: because the vocabulary is tiny, the whole embedding table is only 98 × 512 ≈ 50k parameters — 0.15% of the model. (An LLM with a 100k-token vocabulary spends billions here. Modeling characters directly is part of why this model can be small.)

---

## Part II — The model

### §4 Attention, from scratch

After embedding, each cell knows only its own character. To draw coherently, cells must exchange information: the cell continuing a stroke must know where the stroke is heading; the left half of a face constrains the right. The mechanism is **attention**.

**The idea.** Attention is a soft, learned lookup. Every cell asks a question about the rest of the grid, every cell advertises what it contains, and each cell receives a blend of other cells' content weighted by how well their advertisements match its question. Three learned linear maps produce, from each cell's vector $x$, a **query** $q$ (what am I looking for?), a **key** $k$ (what do I contain, as seen by searchers?), and a **value** $v$ (what content do I hand over if selected?):

$$q = W_Q x, \qquad k = W_K x, \qquad v = W_V x \tag{4.1}$$

Stack all 3,840 cells' queries into a matrix $Q$ (likewise $K$, $V$). The whole exchange is one formula — **scaled dot-product attention**:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V \tag{4.2}$$

Read it inside-out: $QK^\top$ computes every query's dot product with every key — a 3,840×3,840 table of match scores. Dividing by $\sqrt{d_k}$ (the key dimension, 64) keeps scores in a range where softmax doesn't saturate. The row-wise softmax turns each cell's scores into weights summing to 1, and multiplying by $V$ gives each cell its personalized weighted average of everyone's values. Crucially, $W_Q, W_K, W_V$ are *learned* — the network learns what is worth asking and advertising.

**Multiple heads.** One attention pattern per layer is a bottleneck: a cell might simultaneously need "where does my stroke continue?" and "how dense is ink in my region?". So the 512 dimensions split into **8 heads** of 64, each running (4.2) independently with its own projections; outputs are concatenated and mixed by one more linear map. Eight parallel conversations instead of one.

**The rest of the block.** Attention only mixes information; each cell also needs private processing capacity. Every block follows attention with a **feed-forward network** applied identically per cell:

$$\operatorname{FFN}(x) = W_2 \, \operatorname{GELU}(W_1 x), \qquad 512 \to 2048 \to 512 \tag{4.3}$$

Two further ingredients make deep stacks trainable. **Residual connections**: each sublayer computes a *correction* added to its input, $x \leftarrow x + \operatorname{Sublayer}(x)$, so gradients flow straight through the additions and each layer only learns a small refinement. **Layer normalization** rescales each cell's vector to standardized statistics before each sublayer ("pre-norm", the modern stable arrangement).

The full model stacks **8 such blocks** (one extra sublayer per block — cross-attention to the prompt — arrives in §6). After the last block: LayerNorm, then a linear map 512 → 98 producing each cell's logits.

```
            input grid  48×80 tokens
                 │  embedding lookup (§3) + 2D RoPE positions (§5)
                 ▼
   ┌────────── block × 8 ────────────────────────────┐
   │ x ← x + SelfAttention(LN(x))      8 heads, §4   │
   │ x ← x + CrossAttention(LN(x), caption)     §6   │
   │ x ← x + FFN(LN(x))                512→2048→512  │
   └─────────────────────────────────────────────────┘
                 │  LayerNorm → Linear 512→98
                 ▼
        per-cell logits  48×80×98  →  softmax → probabilities
```

**Why bidirectional.** Language models like GPT use *causal* attention: each token may only look left, because they generate text one word at a time. This model's attention is **unrestricted (bidirectional)**: every cell sees every other cell. A picture has no reading order — the hull of a boat constrains the sail above it as much as the reverse — and, as §10 shows, decoding never needs a past/future distinction: it always predicts *masked* cells from *all visible* cells. This is BERT's shape (hence "ASCIIBert"), not GPT's.

**Parameter budget:** embeddings 0.05M · 8 transformer blocks 33.6M (of which cross-attention 8.4M) · conditioning module 0.8M · output head 0.05M → **34.5M total**.

### §5 Where am I? Positions and 2D RoPE

Formula (4.2) has a blind spot: it treats the grid as a *bag* of cells — shuffle them and the attention scores shuffle identically. Position must be injected explicitly.

The naive fix (a learned vector per position, added at the input) has two flaws: positions learn *independent* vectors, so a stroke pattern learned at one location doesn't transfer to another; and there is no sane value for positions never seen in training, so the model can't run on larger canvases.

**Rotary position embeddings (RoPE)** encode position by *rotating* the query and key vectors. Pair up the dimensions of $q$ and treat each pair as a point in a plane; at grid position $m$, rotate pair $i$ by angle $m\theta_i$:

$$R(m\theta)\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i \cdot q_{2i} - \sin m\theta_i \cdot q_{2i+1} \\ \sin m\theta_i \cdot q_{2i} + \cos m\theta_i \cdot q_{2i+1} \end{pmatrix} \tag{5.1}$$

with a frequency spectrum $\theta_i = 10000^{-2i/d}$ (fast-spinning pairs resolve nearby offsets, slow ones resolve far). The magic is in the dot product: for a query rotated by $m\theta$ and a key rotated by $n\theta$,

$$\langle R(m\theta)q, \; R(n\theta)k \rangle = \langle R((m-n)\theta)q, \; k \rangle \tag{5.2}$$

— the attention score depends only on the **relative offset** $m - n$, never on absolute location. (Rotate two clock hands and compare: only the angle *between* them survives.)

**Making it 2D:** a grid position is (row, column), so the head dimensions split in half — one half rotates by the *row* index, the other by the *column* index. Attention then depends on $(\Delta\text{row}, \Delta\text{col})$: true 2D relative geometry. Two consequences for free:

- **Translation equivariance** — "the cell one row down and two columns right continues my diagonal" is the same computation anywhere. Shape knowledge is learned once, not per location.
- **Extrapolation** — rotations are defined by formula for any index, so the model runs on canvases up to 96×160 despite training at 48×80. The site's upscaler exploits exactly this: anchor the original characters at strided positions on a double-size canvas and let the model inpaint between them.

### §6 Listening to the prompt: CLIP and cross-attention

**CLIP in brief.** CLIP is a pair of encoders — one for images, one for text — trained on ~400M (image, caption) pairs with a **contrastive** objective: embed both sides, push each image's vector toward its own caption's and away from the batch's other captions. Similarity is **cosine similarity** — the angle between vectors:

$$\operatorname{sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert} \tag{6.1}$$

This forces a shared geometry where "a goat", "a billy goat", and photographs of goats land near each other. We use CLIP's *text* encoder, **frozen** (never trained further): each prompt becomes up to 24 token vectors of 512 dims plus one pooled summary vector. Freezing means our small dataset cannot damage the imported knowledge — and it is why the model generalizes to prompt words it never saw in training captions: unseen-but-related words arrive pre-positioned near seen ones.

**Cross-attention: the injection point.** Every block gets a second attention sublayer where **queries come from grid cells but keys and values come from caption tokens**:

$$\operatorname{CrossAttn}(x, c) = \operatorname{softmax}\!\left(\frac{(W_Q x)(W_K c)^\top}{\sqrt{d_k}}\right)(W_V c) \tag{6.2}$$

Each cell individually asks the prompt "which of your words concern me?" — a hull cell can attend to "sailboat" while another region attends to "two". Keeping the caption as 24 separate tokens (not one squashed vector) is what preserves this compositional addressability. No RoPE here: word order is already baked into CLIP's output, and grid-to-text offsets are meaningless.

Three engineering details repay study:

- **Zero-initialized output projections.** Each cross-attention sublayer's final linear map starts all-zero, so at initialization the sublayer outputs exactly 0 and the residual stream is untouched — an exact no-op. Gradients still flow (a zero *weight* does not mean a zero *gradient*), so the pathway fades in with training. Operationally this allowed splicing text conditioning into a checkpoint trained *without* it and resuming undisturbed — no geometry retrain on a fixed GPU budget. (This is the ControlNet trick applied to conditioning.)
- **A learned null token.** Uncaptioned samples attend to a learned "no prompt" embedding instead of real text. The model thereby learns an explicit unconditional mode — which §11 turns into the second branch of classifier-free guidance.
- **Mask-ratio conditioning.** A small MLP maps the current fraction of masked cells into the global conditioning vector. The model always knows how far along the decode is: early (mostly masked — commit broad layout) versus late (mostly committed — refine details). The analogue of a diffusion model being told its noise level.

---

## Part III — Training

### §7 The objective: masked prediction

How do you teach a network to draw? Not by asking it to draw — by asking it to **restore**. Take a real grid $x$, choose a random mask ratio $r$, replace a random $r$-fraction of cells with `[MASK]`, and train the network to predict the original character at every masked position:

$$\tilde{x} = \operatorname{corrupt}(x, r), \qquad \text{model outputs } p_\theta(x_i \mid \tilde{x}, c) \text{ for masked } i \tag{7.1}$$

This is self-supervised: the data provides its own labels. To restore well, the network must learn everything we actually want — stroke continuity, symmetry, object structure, and (because the caption $c$ is provided) which shapes go with which words.

**Cross-entropy.** Wrongness at one cell is the negative log of the probability assigned to the true character:

$$\ell_i = -\log p_\theta(x_i \mid \tilde{x}, c) \tag{7.2}$$

Assign probability 1 to the truth → loss 0; assign 0.01 → loss ≈ 4.6. Minimizing (7.2) in expectation maximizes the likelihood of the data — the statistically principled way to fit a distribution. The batch loss averages over masked cells only.

**Three modifications, each earned by a failure:**

**(a) Space down-weighting.** ~80% of target cells are the space character. Under plain averaging, predicting space everywhere is a low-loss strategy — the statistical prior behind blank collapse (§14.1). Cells whose true character is space get weight $w = 0.4$ in a weighted mean:

$$L = \frac{\sum_i w_i \ell_i}{\sum_i w_i}, \qquad w_i = 0.4 \text{ if } x_i = \text{space, else } 1 \tag{7.3}$$

(Dividing by $\sum w_i$, not the count, keeps the loss scale independent of how much space a batch contains. As of v3 the weight is no longer the hand-tuned 0.4 but auto-calibrated per stage from the dataset's space fraction — Appendix C.1.)

**(b) Glyph-aware label smoothing.** Plain CE says predicting `|` where `/` belongs is exactly as wrong as predicting `@` there — but to the eye it isn't. The target softens from one-hot toward a visual-similarity distribution $S$ over glyphs (computed from rendered glyph bitmaps), with $\epsilon = 0.1$:

$$y^\text{soft} = (1 - \epsilon) \cdot \operatorname{onehot}(x_i) + \epsilon \cdot S_{x_i} \tag{7.4}$$

**(c) The mask-ratio distribution.** Generation (§10) *starts* at $r = 1$ and sweeps down to 0 — so training must cover high ratios well, or the first and most important decode steps run on inputs the model has never seen. Ratios are drawn by warping a uniform variable through a sine:

$$u \sim \operatorname{Uniform}(0,1), \qquad r = r_\text{min} + (r_\text{max} - r_\text{min}) \cdot \sin(\pi u / 2) \tag{7.5}$$

with $r_\text{min} = 0.15$, $r_\text{max} = 1.0$. The sine's flattening near 1 concentrates samples at high mask ratios — where generation is hardest — and the maximum of exactly 1.0 guarantees the blank canvas itself is in-distribution.

Finally, with probability 0.1 the caption is dropped and replaced by the null token (`COND_DROPOUT`) — the training-side requirement of classifier-free guidance (§11).

(v3 adds a fourth modification with a story of its own — **self-context training**, which rebuilds part of the visible context from the model's own sampled predictions to close the exposure-bias gap between training and decoding. It is the subject of Appendix C.3, including the first implementation that taught the model exactly the wrong lesson.)

### §8 Optimization machinery

| Mechanism | Setting | Why |
|---|---|---|
| AdamW | lr 3e-4, wd 0.01 | per-parameter adaptive steps; weight decay regularizes |
| Warmup | 500 steps | early gradients are garbage; ramp the LR from 0 |
| Cosine LR decay | per stage | big steps early to travel, small steps late to settle |
| Gradient clipping | max norm 1.0 | one bad batch can't launch the weights into space |
| bf16 autocast | on CUDA | ~2× faster; bf16's wide exponent range stays stable |
| DDP | multi-GPU | per-GPU gradient shards averaged every step |
| LR × world size | linear scaling | N GPUs = N× batch = N× fewer steps; larger steps compensate |
| Dropout | 0.0 (off, as of v3) | at 0.1 it silently vetoed prompt following — Appendix C.2; the random masking itself regularizes |

**Exponential moving average (EMA).** Late in training the weights jitter around the loss basin. A second copy, updated as a slow decay toward the live weights, averages the jitter out:

$$\bar\theta \leftarrow \lambda \bar\theta + (1 - \lambda)\theta, \qquad \lambda = 0.999 \tag{8.1}$$

Effective averaging horizon: $1/(1-\lambda) = 1{,}000$ recent steps. The EMA copy generates consistently cleaner samples and ships as `final_model.pt`; checkpoints store the live weights (matching the optimizer state for resume) with EMA alongside.

### §9 The three-stage curriculum

| Stage | Data | Epochs | LR | Teaches |
|---|---|---|---|---|
| 1 · geometry | 200k procedural grids (lines, rectangles, circles, crosses) with exact captions | 6 | 3e-4 | the glyph vocabulary of strokes |
| 2 · shading | synthetic engine output + sketch conversions + geometry replay | 6 | 1e-4 | real subjects; prompt following |
| 3 · human | ~5k scraped human pieces (BLIP-captioned) + shading & geometry replay | 3 | 5e-5 | how humans actually deploy characters |

**Replay against forgetting.** Neural networks *catastrophically forget*: fine-tuning on stage 3's 5k pieces would overwrite prompt-following learned from 100k+ captioned samples, because nothing in the gradient asks to preserve it. Each stage mixes in samples from earlier stages to keep old skills inside the gradient. Replay subsets use a fixed seed so every DDP rank draws the identical mix. (v1 hard lesson: skills must be replayed in *every* later stage, at a non-trivial fraction — see §14 and the Appendix.)

**Decreasing learning rates** (3e-4 → 1e-4 → 5e-5): later stages refine an already-good model; big steps would trample earlier learning. The zero-init conditioning pathway (§6) is what let stages splice across an architecture change without restarting.

---

## Part IV — Generation

### §10 MaskGIT decoding, step by step

Training taught one skill: *given a partially visible grid, predict every hidden cell*. Generation bootstraps a full picture out of that skill, starting from nothing.

Predicting all 3,840 cells from the blank canvas at once fails: with no context, cells are predicted *independently*, and independent guesses don't compose (each cell might belong to a different imagined goat). Committing one cell at a time is coherent but needs 3,840 sequential passes. MaskGIT is the middle path: **commit only what you're confident about, then look again.**

**Algorithm** (as implemented; $T = 10$ steps):

1. Start with all $N$ cells = `[MASK]`. For $t = 1 \dots T$:
2. Forward pass (with CFG, §11) → logits for every cell; apply temperature (2.3) and the space bias (10.3); softmax.
3. For every still-masked cell, *sample* a candidate $\hat{x}_i$ and record its probability $p_i$.
4. Score confidences with annealed exploration noise (10.2).
5. The schedule (10.1) says how many cells must remain masked; commit the top-confidence candidates down to that number, re-mask the rest.
6. After the loop: 2 revision passes — re-mask the 10% of committed cells the model is now *least* confident about and refill them with a short version of the same loop.

**The schedule.** Cells allowed to remain masked after step $t$ follow a cosine:

$$m(t) = \lfloor N \cdot \cos(\pi t / 2T) \rfloor \tag{10.1}$$

Cosine starts flat and steepens: early steps commit *few* cells (context is thin, mistakes get locked in), late steps commit *many* (the picture is mostly decided). At $t = T$ the target is 0.

**Confidence, with exploration noise:**

$$\operatorname{conf}_i = \log p_i + \kappa\,(1 - t/T) \cdot g_i, \qquad g_i = -\log(-\log u_i), \; u_i \sim \operatorname{Uniform}(0,1) \tag{10.2}$$

$g_i$ is **Gumbel** noise — the natural randomness for perturbing log-probabilities. The annealing factor $(1 - t/T)$ means: explore early (don't always commit the same safest cells first, or every sample looks alike), commit strictly by confidence at the end. $\kappa = 0$ gives a deterministic decode — and probing showed the two settings serve different masters: exploration noise buys sample diversity but actively hurts *precise structure* (a scattered commit order breaks up clean strokes), so geometric subjects decode best at $\kappa = 0$.

**The space bias** — a decode-time counterweight to blank collapse (§14.1): the space logit is penalized in proportion to how empty the grid still is,

$$z_\text{space} \leftarrow z_\text{space} - b \cdot \rho_t, \qquad \rho_t = \text{current fraction masked} \tag{10.3}$$

Full strength on the empty canvas, zero once mostly committed.

**Why this beats the alternatives:** ten forward passes instead of 3,840 (~400× cheaper than autoregressive); every commitment made *with context* (unlike single-shot); early mistakes get one more chance via revision. The intermediate grids are exactly the frames the website's materialize animation replays — the visualization *is* the algorithm.

One more gift: cells that are not masked are simply never touched, so **inpainting and upscaling need no new machinery**. Fix any cells (e.g. anchor a finished 48×80 piece at every other position of a 96×160 canvas), mask the rest, run the same loop — the model fills gaps consistent with the anchors, at a resolution it never trained on (courtesy of relative positions, §5).

### §11 Classifier-free guidance

A conditioned model often follows its prompt too weakly — the caption is one input among thousands of cells, and the path of least resistance is to draw *something* generic. Guidance is the standard amplifier.

Run the model twice per step — once with the prompt, once with the null token (the unconditional mode learned via the 10% caption dropout) — and extrapolate the logits *past* the conditional prediction, away from the unconditional one:

$$z_\text{guided} = z_\text{uncond} + s \cdot (z_\text{cond} - z_\text{uncond}) \tag{11.1}$$

$s = 1$ is the plain conditional model; $s > 1$ amplifies every way the prompt changes the prediction. Justification via Bayes: since $\log p(c|x) = \log p(x|c) - \log p(x) + \text{const}$, the difference $(z_\text{cond} - z_\text{uncond})$ behaves like an *implicit classifier's* opinion of "how much does this look like the prompt?" — and (11.1) samples from a distribution re-tilted by that opinion, $p(x|c)\,p(c|x)^{s-1}$. No separate classifier is ever trained — hence "classifier-free."

Implementation: the two passes run as one batch of two rows, so guidance costs about half a pass of extra latency, not double.

> **An empirical surprise.** In this domain, guidance acts as an **object-count and literalness dial**: "a cross" at $s = 3$ produces several crosses; $s \approx 1.5$–2 yields one clean subject. Why: the strongest direction in which "more prompt-like" can grow is *more instances of the subject's features*. Discovered by cheap probe sweeps; now the UI default. A knob's textbook meaning does not automatically transfer to a new domain. (The v2 "rise" guidance schedule attacks the flood regime directly — see the Appendix.)

### §12 Best-of-k and re-ranking

Sampling is stochastic and cheap (~10 forward passes), so the server draws $k$ variations and lets a judge pick. The judge is CLIP again, both encoders this time: render each candidate grid to a real image through the glyph atlas (each character's 12×8 bitmap), embed the rendering with CLIP's image encoder and the prompt with its text encoder, score with cosine similarity (6.1), return best-first.

A pure inference-time quality win — *judging* pictures is easier than *making* them, and we import a strong judge for free.

A door deliberately left open: the glyph rendering is implemented differentiably (probabilities × glyph bitmaps, summed — an expectation over rendered images), so a *training* loss can backpropagate through the rendering. (As of v2, it does — see the Appendix.)

---

## Part V — Data

### §13 The synthetic data engine

The central obstacle: **captioned ASCII art does not exist at scale.** Scraping yields a few thousand pieces, mostly uncaptioned, stylistically chaotic — enough to teach *style* (stage 3), hopeless for *prompt following*, which needs six figures of (caption, grid) pairs. The engine manufactures them, inverting the usual pipeline: **choose the caption first, then fabricate artwork for it** — every label correct by construction.

```
caption ──► SD-Turbo ──► trim 4% ──► Otsu ──► glyph ──► ink band ──► CLIP ──► (grid,
 bank        txt2img      border      binarize  match     2–55%       filter    caption)
 627 subj    4 steps                            6-D                   ≥ 0.2
 75/15/10    guidance 0
```

Stage by stage, with the reason each exists:

- **Prompt bank.** Hundreds of subjects expanded through templates into three categories — singles (75%), counted subjects like "two fish" (15%), pairs (10%). The mix is enforced explicitly because a naive combinatorial bank is $n^2$ pairs versus $n$ singles — it came out 93% pairs (§14.3). Pools cycle when exhausted rather than demanding uniqueness (a uniqueness requirement once hung the engine in an infinite loop).
- **SD-Turbo.** A distilled text-to-image model producing an image in 4 steps (guidance 0 — Turbo models don't use CFG). Prompts wrap in an icon-style template because raw outputs are shaded and textured, which converts to unreadable mush. (v2 adds outline and tonal styles — Appendix.)
- **Border trim (4%).** SD-Turbo letterboxing produced solid sidebars; untrimmed, the dataset would teach the model to draw frames around everything.
- **Otsu binarization.** The grayscale image thresholds into ink/paper at the split maximizing between-class variance:

$$t^* = \arg\max_t \; \omega_0(t)\,\omega_1(t)\,[\mu_0(t) - \mu_1(t)]^2 \tag{13.1}$$

  where $\omega_0, \omega_1$ are the two classes' pixel fractions and $\mu_0, \mu_1$ their mean intensities — the natural valley between a bimodal image's ink and paper modes. (The first attempt — a fixed percentile threshold — forced an ink *quota* and turned photos into blobs.)
- **Glyph matching (the converter).** Deterministic image → ASCII: each cell's patch is summarized by a 6-dimensional shape vector describing where its ink mass lies, and the chosen character minimizes the distance to the glyph's own vector:

$$\operatorname{char}(\text{cell}) = \arg\min_{g \in \text{glyphs}} \lVert \phi(\text{cell}) - \phi(g) \rVert^2 \tag{13.2}$$

  The same converter powers the site's image-intake tab — the model learns, and the site speaks, one dialect of ASCII. (Its full pipeline — adaptive gamma → CLAHE → Sobel edge blend — is *tonal*; v1's binarize step was discarding that capability. See the Appendix.)
- **Ink-fraction band** [0.02, 0.55]: rejects blank and solid-fill failures before any expensive scoring.
- **CLIP consistency filter.** Embed the generated *image* and its own caption; drop pairs below 0.2 cosine similarity — closing caption-first synthesis's one failure mode (the image not depicting its prompt).

**Throughput** is a producer/consumer pipeline: the GPU generates batch $N{+}1$ while a CPU worker pool converts batch $N$ (spawn context, workers return plain numpy buffers — §14.5); a test proves the pooled path byte-identical to serial. Uncaptioned grids (scraped art) are auto-captioned with BLIP.

> **The core idea.** The engine converts a data-scarcity problem into a compute problem — and compute is rentable by the hour. The engine is 20% generation and 80% quality control.

---

## Part VI — Lessons

### §14 Six failures and their fixes

**14.1 Blank-grid collapse.**
*Symptom:* reconstruction fine; free generation empty.
*Diagnosis:* three compounding causes. (i) The prior: ~80% of cells are space, so CE makes space the minimum-risk guess under uncertainty. (ii) The decoding loop turns the prior into a trap: MaskGIT commits the most *confident* cells first — on an empty canvas, the spaces — and committed blank becomes the context everything later conditions on. A feedback loop. (iii) Distribution gap: training rarely sampled mask ratios near 1.0, but generation always *starts* at exactly 1.0.
*Fix:* all three levels at once — space reweighting in the loss (7.3), sine-biased ratio sampling to 1.0 (7.5), annealed decode-time space bias (10.3).
*Lesson:* class imbalance + confidence-ordered commitment is a feedback loop, not just a bias. And always verify training covers the exact state inference starts from.

**14.2 Guidance turned out to be an object-count dial.**
*Symptom:* "a cross" at guidance 3 → several crosses; raising guidance added clutter, not fidelity.
*Diagnosis:* (11.1) amplifies every direction in which conditional differs from unconditional; here the largest is *more instances of the subject's features*.
*Fix:* calibration, not code — probe sweeps put the sweet spot at ~1.5–2, now the UI default, surfaced as a composition-density dial.
*Lesson:* a knob's textbook meaning is a hypothesis in a new domain. Cheap empirical sweeps beat assumptions.

**14.3 Chimera images from paired prompts.**
*Symptom:* "X and Y" prompts rendered fused monsters (goat-boats); the bank was 93% such prompts.
*Diagnosis:* combinatorics ($n^2$ pairs vs $n$ singles dominated silently) × few-step t2i models fusing multi-subject prompts.
*Fix:* explicit 75/15/10 category mix with per-category pools and cycling.
*Lesson:* every dataset property you don't control explicitly is being set implicitly — usually by combinatorics. Look at the actual samples.

**14.4 Multi-GPU crash from a conditionally-used parameter.**
*Symptom:* DDP crashed in the gradient reducer; single-GPU fine.
*Diagnosis:* the learned null token only entered the forward graph when a batch contained an uncaptioned sample; DDP needs every parameter in every step on every rank.
*Fix:* unconditional `torch.where` selection — the null token is always in the graph, unused rows contribute zero gradient. Proven by a 2-process CPU smoke test now living in the suite.
*Lesson:* data-dependent control flow over *parameters* is a distributed-training landmine. Test distributed paths cheaply on CPU before renting hardware.

**14.5 Worker pool death by file-descriptor exhaustion.**
*Symptom:* hours into a run, `received 0 items of ancdata`; pool dead.
*Diagnosis:* workers returned torch tensors; PyTorch ships tensors across processes via file-descriptor sharing, leaking fds under a steady stream of results.
*Fix:* workers return plain numpy uint8 (a 48×80 grid is 3.8 KB); spawn not fork (the parent holds a CUDA context).
*Lesson:* the convenient type for computation is not the safe type for transport. Long runs surface leaks short tests never will.

**14.6 The engine rejecting 90% of its own output.**
*Symptom:* 15:1 rejection, ~0.7 accepted img/s from a GPU generating 6/s; rejection logs spiked at *exactly* 0.900 ink.
*Diagnosis:* SD-Turbo shaded despite prompts; the 0.900 spike was letterboxed square images whose sidebars converted to solid columns; the original percentile binarizer blobified the rest.
*Fix:* the full intake chain of §13 — icon prompt, Otsu, trim, ink band, CLIP filter — each tuned on small preview batches. Keep rate ~10% → ~78%.
*Lesson:* log *why* rejects happen, not just how many — the 0.900 spike was the entire diagnosis. Tune data pipelines on cheap previews before paying for the full run.

### §15 Key decisions, defended

**MaskGIT over autoregressive or diffusion.** Autoregressive needs 3,840 sequential passes and imposes a fictitious reading order on a 2D artifact. Continuous diffusion fights the discreteness of the output (a cell is `/` or `\`, never 0.6 of each). Masked parallel decoding is discrete-native, bidirectional, ~400× cheaper — and its intermediate states handed the site its signature animation for free.

**Characters as tokens, not pixels.** Generate-then-convert caps quality at the converter's expressiveness and can never show character-level *intent* — using `()` for a curve because a human would. Native token generation makes the human fine-tune meaningful; glyph label smoothing (7.4) bridges the discrete loss to visual reality.

**Train from scratch at 34.5M.** Nothing pretrained speaks 48×80 character grids; adapting a giant means paying for billions of irrelevant parameters at inference forever. The domain is narrow enough to saturate; 140 MB runs on CPU and retrains on one rented GPU. Evidence the size suffices: geometry-stage loss flatlined by epoch 2 — the bottleneck is data, not capacity.

**Frozen CLIP + per-block cross-attention.** Frozen CLIP imports a semantic text space for free ("a hound" lands near "a dog" without either in training captions) and freezing protects it from the small dataset. Token-level cross-attention keeps words separately addressable; class-index conditioning was the original design, replaced because it cannot generalize to free text.

**Manufacture the dataset (caption-first), don't scrape.** The scrapeable corpus is thousands of pieces, mostly uncaptioned — sufficient for stage-3 style, hopeless for prompt following. Caption-first makes labels correct by construction; the CLIP filter closes the remaining failure mode. Against "you just distilled SD-Turbo": deliberately, and only for subject *shape* — the converter re-expresses everything in the glyph dialect, stage 3 restores human style, and no diffusion weights exist at inference.

**Three-stage curriculum with replay.** Stages are ordered by abstraction — strokes, subjects, style — each assuming the last. Uniformly mixing 5k precious human pieces into 500k synthetic samples would drown them; a dedicated low-LR stage with replay preserves both.

**EMA weights for the shipped model.** The 0.999 EMA averages ~1,000 steps of late-training jitter and samples consistently cleaner. One extra weight copy during training, nothing at inference.

**Best-of-k with CLIP re-ranking at serve time.** Generation is cheap and stochastic; judging is cheap and reliable. $k$ samples + a free imported judge = pure quality win with zero training cost.

### §16 Glossary

| Term | Meaning |
|---|---|
| token | one symbol from the fixed vocabulary; here, one character in one cell |
| logits | raw pre-probability scores, one per vocabulary symbol |
| softmax | converts logits to a probability distribution (2.2) |
| temperature | divisor on logits controlling sampling randomness (2.3) |
| embedding | learned vector representing a discrete symbol (§3) |
| attention | learned, content-based weighted averaging between positions (4.2) |
| Q / K / V | query, key, value — the three learned projections feeding attention |
| head | one of several independent attention patterns per layer |
| FFN | per-position two-layer network inside each block (4.3) |
| residual stream | the running vector each sublayer adds its correction to |
| LayerNorm | per-vector standardization before each sublayer |
| RoPE | rotary position embedding — position via rotation, so attention sees only relative offsets (§5) |
| cross-attention | queries from the grid, keys/values from the caption (6.2) |
| CLIP | contrastively trained image/text encoder pair; used frozen as text encoder, filter, and judge |
| cross-entropy | −log probability assigned to the true symbol (7.2) |
| label smoothing | softening one-hot targets; here toward visually similar glyphs (7.4) |
| mask ratio | fraction of cells hidden; the model's "noise level" |
| MaskGIT | generation by iterative parallel unmasking, confidence-ordered, cosine schedule (§10) |
| Gumbel noise | natural randomness for perturbing log-probabilities; annealed across decode steps (10.2) |
| CFG | classifier-free guidance — logit extrapolation away from the unconditional prediction (11.1) |
| EMA | slow-moving average of weights used for the final model (8.1) |
| DDP | DistributedDataParallel — multi-GPU training with per-step gradient averaging |
| Otsu's method | automatic binarization threshold maximizing between-class variance (13.1) |
| catastrophic forgetting | loss of earlier skills when training on new data; countered by replay (§9) |
| exposure bias | train/test mismatch: training shows only ground-truth context, decoding conditions on the model's own imperfect commits (Appendix C.3) |
| self-context / SUNDAE | training on context rebuilt from the model's own samples, with truth targets on every originally-masked cell — commits included (Appendix C.3) |
| perceptual loss | auxiliary MSE between differentiably rendered prediction and target (A.2) |
| revision passes | post-fill rounds that re-mask the least-confident committed cells and refill them (§10) |

### §17 Whiteboard self-test

Work these with the answers hidden. When you can do them cold — including writing the equations — the project is yours in the sense that matters.

1. **Walk through one generation end to end, citing each formula used.**
   Prompt → frozen CLIP → 24 token vectors + pooled vector (§6). Grid ← 3,840 × `[MASK]`. For t = 1…10: batched forward (conditional + null rows), CFG-combine logits (11.1); temperature (2.3); space bias (10.3); softmax (2.2); sample every masked cell; confidence = log p + annealed Gumbel (10.2); cosine schedule (10.1) fixes how many stay masked; commit top confidence. Then 2 revision rounds. Repeat for k seeds; CLIP re-rank by cosine similarity (6.1); return best first.

2. **Derive why blank collapse happens and why the fix needs all three levels.**
   Prior: 80% of targets are space, so minimizing (7.2) makes space the safe answer under uncertainty. Dynamics: MaskGIT commits the most confident cells first — the spaces — and committed blank becomes the context for later steps: self-reinforcing. Distribution gap: generation starts at ratio exactly 1.0. Loss reweighting (7.3) changes the belief; sine-biased sampling to 1.0 (7.5) closes the gap; the annealed decode bias (10.3) counters the commit-order trap where it operates. Any one alone under-corrects.

3. **Why bidirectional attention? What breaks with a causal mask?**
   Pictures have no reading order — constraints run in every direction — and masked prediction needs each masked cell to see *all* visible cells. A causal mask blinds cells to half their context and makes parallel unmasking incoherent.

4. **State the RoPE property (5.2) and its two practical consequences here.**
   ⟨R(mθ)q, R(nθ)k⟩ = ⟨R((m−n)θ)q, k⟩ — attention depends only on relative offset; split across two axes it gives (Δrow, Δcol) geometry. Consequences: translation equivariance (stroke patterns learned once apply anywhere) and extrapolation to unseen canvas sizes (up to 96×160), enabling anchored upscaling.

5. **Explain CFG from Bayes' rule, the training requirement, and this domain's surprise.**
   log p(c|x) = log p(x|c) − log p(x) + const, so z_cond − z_uncond acts as an implicit classifier's gradient of prompt-likeness; (11.1) samples from p(x|c)·p(c|x)^(s−1). Training must provide the unconditional branch: captions drop to the learned null token 10% of the time. Surprise: s behaves as an object-count/literalness dial; sweet spot ≈ 1.5–2, found by probe sweeps.

6. **Why is the mask-ratio distribution sine-warped with max exactly 1.0? Write it.**
   r = r_min + (r_max − r_min)·sin(πu/2), u ~ U(0,1), r_max = 1.0 (7.5). The sine's flattening near 1 concentrates training at high ratios — where generation begins and is hardest — and max 1.0 puts the fully-blank canvas itself in distribution.

7. **What do the zero-initialized projections buy, and why do they still learn?**
   At splice time the new cross-attention sublayers output exactly zero, so a pretrained checkpoint resumes bit-for-bit unchanged — no retrain. They still learn because a zero *weight* does not mean a zero *gradient*: ∂L/∂W depends on the sublayer's inputs and downstream error, both nonzero.

8. **Defend the data engine against "you just distilled SD-Turbo."**
   Partially true by design, and only for subject *shape*: the converter re-expresses every image in the glyph dialect, and stage 3 fine-tunes toward human style. The alternative (scraping) cannot yield 100k+ *captioned* grids at any price. Caption-first makes labels correct by construction; the CLIP filter closes the remaining failure mode. The deployed model is 34.5M parameters with no diffusion weights at inference.

9. **The two nastiest systems bugs, with morals.**
   (1) DDP reducer crash: a parameter that participates only sometimes desynchronizes the reducer — no data-dependent control flow over parameters; select with `torch.where`. (2) Pool death ("0 items of ancdata"): torch tensors cross processes via fd-sharing and leak — return plain buffers over IPC; spawn, not fork, when the parent holds CUDA.

10. **Why is 34.5M parameters enough? What's the actual bottleneck?**
    Tiny vocabulary, strong local structure, bounded canvas — low intrinsic dimensionality. Geometry loss flatlined by epoch 2: capacity wasn't binding; data was. Small buys CPU deployability, fast iteration, cheap retraining. Scale the data engine first.

11. **How do inpainting and upscaling fall out of the decoder for free?**
    The fill loop never touches non-masked cells: fix any cells, mask the rest, run the standard loop. Larger canvases work because RoPE positions are relative and defined by formula for any index.

12. **Where would you take the project next, in order of expected return?**
    (1) More and broader engine data — the demonstrated bottleneck. (2) A perceptual training loss through the differentiable glyph renderer. (3) Canvas-size curriculum via the RoPE headroom. (4) Preference tuning on users' best-of-k picks. *(As of v2, items 1 and 2 are implemented — see the Appendix.)*

13. **Why did dropout 0.1 block prompt following, and what experiment proved it?**
    At high mask ratios (where decode starts) the only route to the answer is the precise caption→content pathway through cross-attention; randomly zeroing 10% of activations makes that route unreliable, so training settles for the robust caption-agnostic texture solution. Proof by differential diagnosis: ask the model to memorize just 64 grids. At dropout 0.1 it stuck at weighted CE ≈ 1.0; the identical model at 0.0 reached 0.034 — same data, same capacity, 30× apart on the regularizer alone. Masking already regularizes (fresh random mask per epoch), so dropout was all cost, no benefit. (Appendix C.2)

14. **Self-context v1 vs v2: same corruption, opposite lessons. Explain.**
    Both rebuild visible context from the model's own sampled commits. v1 scored only the still-masked cells — each batch taught "your own wrong commits are valid context; complete around them," and the probe showed the edge echo *worsened*. v2 keeps every originally-masked cell in the loss: a wrong commit sits visible in the input while its target says otherwise, training the model to contradict and overwrite its own errors (SUNDAE). Bonus: this supervises the output head at visible positions — the exact confidence signal revision passes rank by. The loss mask carries the sign of the lesson, and only a generation probe (not the loss curve) could tell the variants apart. (Appendix C.3)

---

## Appendix A — the v2 revision (training fixes)

Probing the first fully trained model (`final_model.pt`) exposed three problems; each drove a v2 change. This is worth studying as a case of *evidence → mechanism → fix*.

**Evidence 1: everything was a filled silhouette — no internal detail.**
Mechanism: Otsu binarization reduced every training image to a solid two-tone shape, so solid shapes were all the model could learn — even though the converter always had a tonal pipeline (adaptive gamma → CLAHE → Sobel edge blending) that binarize was discarding.
Fix: the engine now draws each batch's **style mode** from a mix — `filled` (v1 dialect), `outline` (morphological edge of the binary mask: ink minus its erosion), and `tonal` (shaded-drawing prompt, *no* binarize, full tonal pipeline → the classic density-ramp look). Captions carry style tags ("a goat, shaded"), so at inference the *prompt* selects the dialect.

**Evidence 2: geometry prompts produced blank at `shading_last` and human-texture noise at `final_model`.**
Mechanism: catastrophic forgetting. 10k geometry replay in a ~300k-sample stage (3%) was dilution, not protection; and stage 3 (10 epochs on 5k human pieces) imprinted its texture statistics onto every prompt the model didn't understand.
Fix: geometry replay into shading raised to 60k (~20%); stage 3 now replays geometry *and* shading (multi-source replay); human epochs cut 10 → 3. Plus a silent-truncation bug fixed: a 100k cap was dropping every merged sample past row 100k.

**Evidence 3: guidance ≥ 2 flooded the canvas with ink.**
Mechanism: the unconditional branch predicts blank, so the CFG direction (z_cond − z_uncond) on an empty canvas is essentially "add ink everywhere"; constant scale amplifies it into the flood/inversion regime.
Fix: a **guidance schedule**. "Rise" (after Muse, which used linearly increasing guidance for masked-transformer decoding) starts at scale 1 on the empty canvas and grows to full scale as cells commit:

$$s_\text{eff}(t) = 1 + (s - 1)(1 - \rho_t) \tag{A.1}$$

where $\rho_t$ is the current masked fraction. Layout forms unamplified; details sharpen under full guidance.

**Plus one addition from the "future work" list:** the perceptual loss now trains. On masked cells, the expected glyph bitmap under the predicted distribution is compared to the true glyph's bitmap in pixel space:

$$L_\text{perc} = \operatorname{MSE}\!\big(\textstyle\sum_v p(v)\, A_v, \; A_{x_i}\big), \qquad L = L_\text{CE} + 0.1\, L_\text{perc} \tag{A.2}$$

where $A_v$ is glyph $v$'s rendered bitmap. Cross-entropy charges every wrong glyph full price; this term charges by visual distance — the criterion ASCII art is actually judged by.

---

## Appendix B — the v3 data engine (FLUX, dialects, composites)

The v2 engine run exposed its own problems within hours of launch; each observation drove a same-day fix. Study this appendix as a compressed case study in *data-engine debugging*: every subsection is observation → mechanism → fix, and every fix shipped with a regression test.

### B.1 Switching the image model to FLUX.1-schnell

**Observation.** With the background fix in place, an A/B preview (24 tonal samples per model) showed SD-Turbo and SDXL-Turbo still producing gray, textured backgrounds and muddy subjects, while FLUX.1-schnell delivered clean isolated pencil drawings with a ~100% filter keep rate — correct counts included ("three satellite dishes" was actually three dishes).

**Mechanism.** Turbo-distilled SD models run at guidance 0 — *no classifier-free guidance means weak prompt adherence*, so "isolated on a plain white background" is a suggestion they half-follow. Every downstream fight (background film, shading-despite-negative-prompt, chimeras) traces to that one property. FLUX-schnell is also CFG-free but its base adherence is in a different class.

**Cost and accommodations.** FLUX needed three engine changes: bf16 (it overflows in fp16), call-kwargs filtered to the pipeline's signature (it has no `negative_prompt` parameter), and CPU offload (12B transformer + T5 encoder ≈ 34 GB, more than a 32 GB consumer GPU). Offloaded throughput measured ~0.6 img/s at batch 16 / 2 steps — roughly a tenth of SD-Turbo. The next two subsections are how that price was paid down.

**Lesson.** The dominant term in synthetic-data quality was the *generator's prompt adherence*, not the converter. Filters can reject bad images, but they can't create good ones.

### B.2 Harvesting all three dialects per image (`--mix all`)

**Observation.** At 0.6 img/s, replacing the SD-era data volume with FLUX would take ~6 days.

**Mechanism.** The expensive step is the t2i forward; ASCII conversion is nearly free. And a *shaded* source image contains the other two dialects within it: Otsu-binarize it for a silhouette, take the mask boundary for an outline.

**Fix.** `--mix all` generates one shaded image and converts it three ways, storing three samples with distinct style tags. Effective throughput tripled (~1.8 samples/s); an all-FLUX dataset became a one-day job. Bonus: the three dialects of each subject are literally the same drawing three ways — the cleanest possible style-conditioning signal.

### B.3 Tone-soften: the bonsai problem

**Observation.** A FLUX bonsai with delicate foliage converted with the leaves as mid-density mush — the wispy strokes and the trunk landed in the same glyph band.

**Mechanism.** The converter's enhancement chain (adaptive gamma → CLAHE → edge blend) exists to rescue murky photos; on an already-clean drawing, CLAHE *amplifies* faint pencil strokes into solid ink before glyph matching.

**Fix.** `tone_soften`: after the enhancement chain, blend the ink map back toward the raw (pre-enhancement) tone:

$$\text{ink} \leftarrow (1 - s)\,\text{ink}_\text{enhanced} + s\,\text{ink}_\text{raw} \tag{B.1}$$

Swept $s \in \{0, 0.4, 0.6, 0.8\}$ on a wispy-foliage analog: $s = 0.6$ cut foliage glyph density 31% with the trunk nearly untouched; $s = 0.8$ began erasing the subject. The tonal dialect converts at 0.6.

### B.4 Background flattening (carried from the v2 run)

The first v2 engine run kept 778 tonal samples out of ~78k attempted (1% vs a 45% target): CLAHE amplified the near-white paper texture of t2i outputs into a canvas-wide film of faint glyphs, sending 99% of tonal conversions over the ink cap. Fix: estimate the background level from the image border and clamp everything within 25 gray levels of it to pure white before conversion — paper maps to space, only the subject inks.

### B.5 Solidify: when all three dialects collapsed into one

**Observation.** The mode-check preview on real FLUX output showed the three `.txt` grids for one subject nearly identical — the outline dialect "didn't work."

**Mechanism.** FLUX draws many subjects as *strokes*, not mass. Otsu keeps strokes as strokes; the morphological edge of a thin stroke is the stroke itself; tonal-of-strokes is lighter strokes. Every dialect of a line is the line. (Tellingly, the one solid region in the test image — a dark house body — *did* hollow correctly.)

**Fix.** `_solidify` for the filled/outline dialects: dilate to seal small contour gaps, flood-fill the background from the image border (at quarter resolution for speed), declare everything the flood can't reach "interior", erode the seal back. A line-drawn birdhouse becomes a true silhouette; its boundary becomes a true contour; tonal keeps the interior detail. Solid sources pass through unchanged; open shapes that leak stay strokes and face the ink filter. ~0.23 s/image.

**Lesson.** Verify the *product of the pipeline*, not the pipeline. Every stage worked as written; the composition was degenerate on the actual input distribution.

### B.6 The tag flip: tonal is the untagged flagship

Originally the untagged caption denoted the filled dialect (to stay merge-compatible with v1 data). Once v3 became pure-FLUX, that convention was backwards: a user typing "a dragon" would get a silhouette, with the flagship look hidden behind ", shaded". v3 flips it — **plain captions mean tonal**; silhouette and outline carry explicit tags (", silhouette", ", outline style"). Consequence: v3 data must not be merged with v1/v2 payloads, where untagged meant filled.

The dialects were kept at all (rather than going tonal-only) for three reasons: they are free byproducts of images already paid for; silhouette and contour are *sub-skills* of shading with unambiguous binary supervision (clean auxiliary gradient for a small model); and the style tags double as a product feature.

### B.7 Composites: counts and pairs correct by construction

**Observation/mechanism.** The two weakest label classes were counted subjects ("two fish" — t2i models flub counts) and pairs ("a goat and an owl" — few-step models fuse them into chimeras; §14.3).

**Fix.** Don't generate them — *manufacture* them. The engine keeps a reservoir of recent single-subject images; with `--composites N`, it tiles one subject 2–3 times (scale/flip/offset jitter) for count samples, or places two different subjects side by side for pair samples. The labels are correct by construction, the images are already paid for, and each composite converts through all three dialects like any other image. This also frees the *generation* prompts to be single-subject only — the regime where the t2i model is most faithful.

**Lesson (general form of B.2 and B.7).** In a synthetic-data engine, ask of every expensive artifact: *what else can be derived from it for free?* One FLUX image ended up yielding up to nine training samples (3 dialects × {itself, a count composite, a pair composite}) — a ~10× improvement in samples per GPU-second over the naive one-image-one-sample design, with *better* labels than generation could provide.

### B.8 Operational scars (so they are not repaid)

- A preview run silently overwrote the full v1 dataset because `--out` defaulted to the real dataset path → the engine now refuses to write over an existing file without `--overwrite`.
- The v1 model checkpoint was deleted on the instance by a broad `rm checkpoints/*.pt` during relaunch prep (recovered from a laptop copy) → checkpoints get copied off the instance the moment a stage completes, and cleanup commands name what they keep.
- A leftover `SYNTH=0` environment variable silently skipped the engine and launched training against the wrong dataset → pipeline stages announce themselves in the log, and launches are verified by watching for those lines, not by absence of errors.

---

## Appendix C — the v3 training run (dropout off, auto space weight, exposure bias & self-context)

The first v3 training attempts on the FLUX dataset produced a fresh crop of failures, and their diagnosis reached deeper into the training recipe than anything before — down to a default hyperparameter and to the objective itself. Same format as Appendix B: observation → mechanism → fix.

### C.1 Auto-calibrated space weight: a constant that was secretly a measurement

**Observation.** On v3 data the blank-collapse symptoms of §14.1 returned despite the space weight of (7.3) being in place.

**Mechanism.** $w = 0.4$ was hand-tuned on v1's ~75%-space data. It is not a universal constant — it is a *measurement of that dataset in disguise*. v3's tonal art is sparser (~90% space), and at that fraction a weight of 0.4 still leaves the space class with ~78% of the total gradient: the attractor reborn.

**Fix.** Compute the weight per training stage from the dataset's measured space fraction $f$, targeting an equal total gradient share for space and ink:

$$w = \frac{1 - f}{f} \tag{C.1}$$

(≈ 0.33 at 75% space, ≈ 0.11 at 90%.) The correction now tracks the data.

**Lesson.** A constant tuned against one dataset silently encodes that dataset's statistics. When the data changes, recompute — or better, derive the constant from the data at run time.

### C.2 Dropout silently vetoed prompt following

**Observation.** Conditioning was verified wired end-to-end (gradients flowing, cross-attention non-zero), yet generations were caption-agnostic texture. The clincher: a sanity test asking the model to simply *memorize 64 grids* stuck at weighted CE ≈ 1.0 — a task 34.5M parameters should crush.

**Mechanism.** Dropout at 0.1 — the transformer default nobody questions. At high mask ratios (where every decode *starts*) there is almost no grid context to lean on; the only route to the right answer runs through the high-precision caption→content pathway. Randomly zeroing 10% of activations makes that precise route unreliable, so gradient descent settles for the robust, caption-agnostic texture solution instead — in exactly the regime generation runs in.

**Fix.** `DROPOUT = 0.0`. The ablation was decisive: the identical model at 0.0 hit loss 0.034 on the memorization task — a 30× gap from a regularizer alone. Overfitting is guarded by the masking itself: every sample is seen through a fresh random mask each epoch, a stronger and better-matched regularizer than dropout.

**Lesson.** Default regularizers are not free — dropout can silently destroy a pathway whose value lies in its precision. And the 64-sample memorization test is the cheapest differential diagnostic there is: it isolates optimization pathology from data and capacity questions.

### C.3 Edge echo — exposure bias, and a fix that backfired

**Observation.** With dropout off and the space weight calibrated, prompted shapes finally appeared — but drawn with duplicated, cascading borders: rectangles nested inside rectangles, diamond interiors filled with dents and thatch, crosses with parallel multi-stroke arms. The artifact persisted through every decode knob: gumbel 0, guidance 1.5, the rise schedule, even 8 revision passes ("they didn't clean up").

**Mechanism.** MaskGIT commits many cells *in parallel* per step, each sampled independently. On a thin context, several offset copies of the same edge commit at once — each locally valid, collectively incoherent. Later steps then face committed duplicates, and here the training gap bites: the model only ever saw *ground-truth* context in training, so its learned response to any plausible visible edge is to **extend** it, never to suppress it. This train/test mismatch — train on truth, decode on your own imperfect commits — is **exposure bias**. Decode knobs cannot fix it; the model lacks the skill, not the opportunity.

**Fix, in two acts.** *Self-context training* (SUNDAE-style step-unrolled denoising): with probability 0.5 per batch, a no-grad forward pass samples the model's own predictions and reveals a random 25–75% of the masked cells as visible context built from those samples; the mask-ratio conditioning is recomputed from the cells still literally `[MASK]`. Cost: one extra no-grad forward on affected batches, ~25% epoch time.

But the first implementation carried a sign error in the *lesson taught*: it scored only the still-masked cells, excluding the self-committed ones from the loss. Training looked perfectly healthy — smooth loss curve, 98.8% accuracy, no regression — yet the A/B probe showed the echo got **worse**: the diamond collapsed to thatch and the cross grew parallel arms. Every batch had taught "your own wrong commits are legitimate context; complete around them."

The corrected objective keeps **every** originally-masked cell in the loss:

$$L = \frac{\sum_{i \in M} w_i \ell_i}{\sum_{i \in M} w_i}, \qquad M = \text{all originally-masked cells, committed or not} \tag{C.2}$$

A wrong commit now sits visible in the input while its target says otherwise, so the model is explicitly trained to contradict and overwrite its own mistakes. This also supervises the output head at visible positions — which is exactly the confidence signal revision passes rank by.

**Act three — the decode-side half of the problem.** The corrected objective helped but didn't finish the job. The decisive evidence came from an *inpainting probe*: shown the left half of a real rectangle, the retrained model completed the pinned region cleanly — and produced a band of competing `|` columns exactly where the unseen right edge could fall. Every pipe a different answer to the one unresolved question. The echo was never ignorance; it was **several correct answers committed simultaneously** — and the cosine schedule (10.1) guarantees it, because the ambiguous low-confidence cells pile up in the final steps and mass-commit in one parallel batch. The finishing move is a **commit cap** (`max_commit`, default 8 cells/step; the fill loops past the scheduled steps until complete), which resolves the ambiguous tail one small batch at a time, each commit becoming context for the next. Confirmed by A/B at identical settings: "a cross" went from 155-ink multi-stroke chaos to a perfect 28-ink cross; the rectangle from 300-ink nested echo to one 97-ink bordered box; capped half-rectangle inpainting produced a single clean closing edge; the capped diamond completion hit 100% accuracy, precision and recall.

**Lesson.** Fixing exposure bias means training the model to *contradict* its own errors, not merely to see them — the loss mask carries the sign of the lesson. Validate training-side fixes with a decode-side probe before scaling up: the failed variant was invisible in every training metric; only generation revealed it taught the opposite of what was intended. And when an artifact survives a training fix, decompose it: this one was two problems wearing one costume — an exposure-bias component (training-side) and a parallel-commit multimodality component (decode-side), each needing its own cure.

### C.4 Curriculum trims that came with the run

Stage 1 (geometry) was cut from 15 epochs to 6 with a full cosine anneal after its loss was observed flat from epoch 2 — the conditioning pathway keeps training through stages 2 and 3 anyway. Together with Appendix A's replay corrections (60k geometry into shading, multi-source replay into the human stage, human epochs 10 → 3), the curriculum now spends compute where the loss curves say it matters.

### C.5 The "triangle" was never a triangle

**Observation.** Spotted by eye in an inpainting probe's ground-truth panel: the grid captioned "a triangle" showed three rays fanning out of one corner — vertical up, diagonal up-right, horizontal right — an open shape that never closes.

**Mechanism.** `draw_triangle` drew the hypotenuse with `/` starting *at the bottom-left corner*, instead of `\` connecting the top of the vertical edge to the right end of the bottom edge. 200k training samples taught the model that fan as "a triangle" — which also explained a decode mystery: triangle completions extended the diagonal indefinitely past the apex, because the learned shape has no closing vertex.

**Fix.** Hypotenuse redrawn `\` from the vertical edge's top to the bottom edge's right end (drawn first so the edges win the corner cells); takes effect on the next data regeneration.

**Lesson.** Synthetic labels are only correct by construction if the construction is correct. Render every primitive and *look at it* — the bug survived three training generations because nobody had reason to inspect the simplest drawing in the codebase.

---

*nASCIIente · study document, first-principles edition · all equations as implemented in the code · revised 2026-07-10*
