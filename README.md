# nASCIIente

A 34.5M-parameter masked transformer that draws ASCII art from a text
prompt. The canvas is 48×80 characters and starts out fully masked. Every
pass predicts all the empty cells at once, keeps the predictions it is most
confident about, and re-masks the rest for the next pass, so a picture
resolves in place instead of being written out in reading order.

Hence the name: *naciente*, Spanish for what is being born, with ASCII in
the middle.

<p align="center">
  <img src="docs/paper/fig/lighthouse.gif" width="428"
       alt='"a lighthouse", an unedited generation from the final model,
            materializing character by character'>
</p>
<p align="center"><em>"a lighthouse", unedited output of the final
model, rendered through its own glyph atlas.</em></p>

**[Read the white paper](https://nrclaudio.github.io/nasciiente/)** for the
architecture, the training recipe, and nine documented failures with their
diagnoses. Two of those are decoding pathologies specific to sparse
discrete canvases, and several are published techniques that did not
survive contact with this domain. Most of what transfers elsewhere is in
that chronicle.

**[Model weights on Hugging Face](https://huggingface.co/nrclaudio/nasciiente-model)**.
Inference is CPU-deployable at ~140 MB, and fast on Apple Silicon via MPS.

## What it is

- The grid of characters is the object being modeled. No pixel image, no
  learned tokenizer: 96 printable glyphs plus `[PAD]` and `[MASK]` are the
  whole vocabulary.
- A bidirectional transformer (8 pre-norm blocks, width 512, factorized 2D
  rotary attention) trained from scratch as a masked predictor, conditioned
  on captions through frozen-CLIP cross-attention.
- Sampled MaskGIT-style: iterative parallel unmasking under classifier-free
  guidance, with a two-regime decode (uncapped exploratory head, then a
  commit-capped sequential tail) that the paper motivates with fixed-seed
  ablations.
- Trained on a self-manufactured captioned corpus: text-to-image
  generations converted deterministically to ASCII in three style dialects,
  every label correct by construction.
- Because the model is bidirectional, the same network does free
  generation, inpainting (fill around fixed characters), and 2×
  super-resolution (anchor characters on a larger canvas, mask the gaps).

## Run the live demo locally

```bash
git clone https://github.com/nrclaudio/nasciiente.git
cd nasciiente
pip install -r requirements.txt

# fetch the checkpoint (or place your own at checkpoints/final_model.pt)
hf download nrclaudio/nasciiente-model final_model.pt --local-dir checkpoints

python -m uvicorn app.server:app --port 8081
```

Open http://localhost:8081, type a prompt, and watch it materialize in the
model's actual commit order. The page also converts uploaded images and
GIFs to ASCII through the training data engine's converter. A `Dockerfile`
and hosting notes live in [`docs/deploy.md`](docs/deploy.md).

## Training your own

The full pipeline (procedural geometry → captioned synthetic subjects →
human ASCII fine-tune, with cross-stage replay) is driven by `main.py` and
documented in §3 and §4 of the paper. One consumer GPU suffices; the v4 run
that produced the released checkpoint was a single rented A100 day.

```bash
python main.py                      # everything: data → three stages
python main.py --stage geometry     # or stage by stage
torchrun --standalone --nproc_per_node=4 training/train.py   # multi-GPU
pytest tests/                       # 130 tests, CPU, ~90 s
```

## Repository map

| Path | What |
|------|------|
| `model/` | ASCIIBert, 2D RoPE, decode loop (tail cap, guidance schedules, critic) |
| `training/` | masking curriculum, self-context training, replay, EMA |
| `data/` | the data engine: t2i → ASCII converter, dialects, quality filters |
| `app/` | FastAPI server + the site (`app/static/index.html`) |
| `docs/paper/` | the white paper (also served via GitHub Pages) |
| `docs/study-guide.md` | first-principles study document of every mechanism |
| `probes/` | the probe outputs behind every claim in the paper |
| `TODO.md` / `STATUS.md` | live project state and queued experiments |

## License

MIT (code). The released weights live in their own
[model repo](https://huggingface.co/nrclaudio/nasciiente-model);
third-party datasets referenced by the data tooling keep their upstream
terms and are not redistributed here.
