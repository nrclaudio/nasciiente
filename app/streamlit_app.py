"""
Streamlit web interface for ASCII Art Transformer.

A terminal-styled studio around the MaskGIT-style model:
  - Generate: prompt-conditioned generation with preset chips, seeds,
    variations, and a step-by-step "materialize" animation
  - Inpaint: fill in the blanks around your own characters, iteratively
  - Guidance lab: one seed swept across guidance scales, side by side
  - Training progress: scrub the per-epoch samples written by train.py
"""

import glob
import os
import sys
import time

import streamlit as st
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE, VOCAB_SIZE,
                    MAX_ROWS, MAX_COLS, CFG_SCALE)
from model.ascii_bert import ASCIIBert
from model.inference import generate, upscale_grid
from data.charset import grid_to_string, MASK_TOKEN
from app.utils import (text_to_partial_grid, count_fixed_positions,
                       grid_to_png_bytes, PNG_SCHEMES)

PRESET_PROMPTS = ["a cat", "a sailboat", "a rocket", "a castle", "a skull",
                  "a diamond", "a rectangle and a cross", "a tree"]

TERMINAL_CSS = """
<style>
.stApp { background: #060c06; }
h1, h2, h3, .stMarkdown, label, .stCaption, p, span, div { color: #9fdc9f; }
[data-testid="stSidebar"] { background: #0a120a; border-right: 1px solid #1d3a1d; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
.stCode, pre, code {
    background: #041004 !important;
    color: #4af626 !important;
    border: 1px solid #1d3a1d !important;
    border-radius: 6px;
    font-size: 11px !important;
    line-height: 1.05 !important;
    text-shadow: 0 0 6px rgba(74, 246, 38, 0.35);
}
.stButton > button, .stDownloadButton > button {
    background: #0a1a0a; color: #4af626; border: 1px solid #2d5a2d;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: #4af626; color: #baffb0;
}
.stTabs [data-baseweb="tab"] { color: #9fdc9f; }
.stTabs [aria-selected="true"] { color: #4af626; }
</style>
"""

BANNER = r"""
   _____  _________ .___.___    _____         __
  /  _  \/   _____/ |   |   |  /  _  \_______/  |_
 /  /_\  \_____  \  |   |   | /  /_\  \_  __ \   __\
/    |    /        \ |   |   |/    |    \  | \/|  |
\____|__ /_______  / |___|___|\____|__  /__|   |__|
        \/       \/                   \/  transformer
"""


@st.cache_resource
def load_model(checkpoint_path, device_str):
    """Load the trained model.

    Tolerates checkpoints from before the current text-conditioning scheme
    (their conditioning modules stay zero-init no-ops); any other mismatch
    fails loudly instead of silently generating noise.

    Returns (model, device, conditioned).
    """
    from model.ascii_bert import load_compatible_state
    device = torch.device(device_str)
    model = ASCIIBert().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt["model_state_dict"] if (isinstance(ckpt, dict)
            and "model_state_dict" in ckpt) else ckpt
    try:
        conditioned = load_compatible_state(model, state)
    except ValueError as e:
        raise ValueError(f"{os.path.basename(checkpoint_path)}: {e}") from e
    model.eval()
    return model, device, conditioned


@st.cache_data
def embed_prompt(text):
    """Embed one prompt (cached so regenerating doesn't rerun the encoder).

    Returns (cond_tokens [L, D], cond_mask [L])."""
    from data.text_embed import embed_captions
    tokens, mask = embed_captions([text])
    return tokens[0], mask[0]


def pick_device():
    """cuda when it exists AND has headroom (don't fight a training run
    for VRAM); cpu otherwise."""
    if torch.cuda.is_available():
        try:
            free, _ = torch.cuda.mem_get_info()
            if free > 3e9:
                return "cuda"
        except Exception:
            pass
    return "cpu"


# Curriculum order for the training-progress browser; stages not listed
# sort after these, alphabetically
STAGE_ORDER = {"geometry": 0, "shading": 1, "human": 2}


def find_checkpoint():
    """Find the best available checkpoint."""
    ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    final = os.path.join(ckpt_dir, "final_model.pt")
    if os.path.exists(final):
        return final
    # Fall back to latest checkpoint
    if os.path.isdir(ckpt_dir):
        pts = sorted([f for f in os.listdir(ckpt_dir) if f.endswith(".pt")])
        if pts:
            return os.path.join(ckpt_dir, pts[-1])
    return None


def animate_steps(placeholder, steps, delay=0.05):
    """Replay the unmasking steps — the grid materializing out of noise."""
    for step_grid in steps:
        placeholder.code(grid_to_string(step_grid), language=None)
        time.sleep(delay)


def show_grid_actions(grid, key, model, device, gen_kwargs, num_steps,
                      temperature, steps=None):
    """Download / animate / upscale controls under a rendered grid."""
    cols = st.columns(4)
    txt = grid_to_string(grid)
    cols[0].download_button("txt", txt, file_name=f"ascii_{key}.txt",
                            key=f"dl_txt_{key}")
    cols[1].download_button("png", grid_to_png_bytes(grid),
                            file_name=f"ascii_{key}.png",
                            key=f"dl_png_{key}")
    if steps and len(steps) > 1:
        if cols[2].button("replay", key=f"anim_{key}",
                          help="Watch it materialize step by step"):
            ph = st.empty()
            animate_steps(ph, steps)
    h, w = grid.shape
    if h * 2 <= MAX_ROWS and w * 2 <= MAX_COLS:
        if cols[3].button(f"upscale ×2", key=f"up_{key}",
                          help="Anchor on a 2x canvas and inpaint the gaps"):
            with st.spinner("Upscaling..."):
                _, up = upscale_grid(model, grid, factor=2,
                                     num_steps=num_steps,
                                     temperature=temperature,
                                     device=device, **gen_kwargs)
            st.code(grid_to_string(up), language=None)
            st.download_button("upscaled txt", grid_to_string(up),
                               file_name=f"ascii_{key}_2x.txt",
                               key=f"dl_up_{key}")
    else:
        cols[3].caption(f"×2 needs ≤{MAX_ROWS}×{MAX_COLS}")


def tab_generate(model, device, conditioned, gh, gw, num_steps, temperature,
                 base_kwargs, seed, randomize):
    # --- Prompt row -----------------------------------------------------
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ""
    if conditioned:
        chip_cols = st.columns(len(PRESET_PROMPTS))
        for col, preset in zip(chip_cols, PRESET_PROMPTS):
            if col.button(preset, key=f"chip_{preset}"):
                st.session_state["prompt"] = preset
        prompt = st.text_input("Prompt (empty = unconditional)",
                               key="prompt",
                               placeholder="describe what to draw ...")
        guidance = st.slider(
            "Guidance", 1.0, 6.0, 2.0, 0.25,
            help="How hard to steer toward the prompt. ~1.5–2 = one clean "
                 "subject; higher = more ink AND more duplicates. 1 = off.")
    else:
        st.caption("This checkpoint has no trained text conditioning — "
                   "generation is unconditional.")
        prompt, guidance = "", 1.0

    n_var = st.radio("Variations", [1, 2, 4], horizontal=True)

    gen_kwargs = dict(base_kwargs)
    if prompt.strip() and conditioned:
        try:
            toks, msk = embed_prompt(prompt.strip())
            gen_kwargs.update(cond_tokens=toks, cond_mask=msk,
                              guidance_scale=guidance)
        except ImportError as e:
            st.warning(str(e))

    if st.button("▷ GENERATE", type="primary", use_container_width=True):
        results = []
        with st.spinner(f"Dreaming in ASCII on {device.type} ..."):
            for i in range(n_var):
                s = int(time.time_ns() % 2**31) if randomize else seed + i
                torch.manual_seed(s)
                t0 = time.time()
                steps, final = generate(model, gh, gw, num_steps=num_steps,
                                        temperature=temperature,
                                        device=device, **gen_kwargs)
                results.append((s, steps, final, time.time() - t0))
        st.session_state["gen_results"] = results
        st.session_state["gen_prompt"] = prompt
        history = st.session_state.setdefault("history", [])
        for s, _, final, _ in results:
            history.append((prompt or "(unconditional)", s, final))
        del history[:-12]  # keep the last 12

    results = st.session_state.get("gen_results", [])
    if results:
        cols = st.columns(min(len(results), 2))
        for i, (s, steps, final, took) in enumerate(results):
            with cols[i % len(cols)]:
                st.code(grid_to_string(final), language=None)
                st.caption(f"seed {s} · {took:.1f}s · {num_steps} steps")
                show_grid_actions(final, f"g{i}", model, device, gen_kwargs,
                                  num_steps, temperature, steps=steps)

    history = st.session_state.get("history", [])
    if history:
        with st.expander(f"Session gallery ({len(history)})"):
            for j, (p, s, g) in enumerate(reversed(history)):
                st.caption(f'"{p}" · seed {s}')
                st.code(grid_to_string(g), language=None)


def tab_inpaint(model, device, conditioned, gh, gw, num_steps, temperature,
                base_kwargs):
    st.write("Type some characters; the model fills everything you leave "
             "blank. Send a result back to the editor to riff on it.")
    default_text = st.session_state.pop("inpaint_seed_text", "")
    user_text = st.text_area("Partial ASCII art", value=default_text,
                             height=260,
                             placeholder="Draw a few strokes ...\n"
                                         "Blank areas get imagined.")
    gen_kwargs = dict(base_kwargs)
    if conditioned:
        hint = st.text_input("Optional prompt to steer the fill",
                             key="inpaint_prompt")
        if hint.strip():
            try:
                toks, msk = embed_prompt(hint.strip())
                gen_kwargs.update(cond_tokens=toks, cond_mask=msk,
                                  guidance_scale=2.0)
            except ImportError as e:
                st.warning(str(e))

    if st.button("▷ FILL IN", type="primary") and user_text.strip():
        partial = text_to_partial_grid(user_text, gh, gw)
        st.caption(f"Fixed positions: {count_fixed_positions(partial)}"
                   f"/{gh * gw}")
        with st.spinner("Inpainting..."):
            t0 = time.time()
            steps, final = generate(model, gh, gw, num_steps=num_steps,
                                    temperature=temperature,
                                    initial_grid=partial, device=device,
                                    **gen_kwargs)
        st.session_state["inpaint_result"] = (steps, final, time.time() - t0)

    if "inpaint_result" in st.session_state:
        steps, final, took = st.session_state["inpaint_result"]
        st.code(grid_to_string(final), language=None)
        st.caption(f"Completed in {took:.1f}s")
        show_grid_actions(final, "inp", model, device, gen_kwargs,
                          num_steps, temperature, steps=steps)
        if st.button("↩ send result to editor"):
            st.session_state["inpaint_seed_text"] = grid_to_string(final)
            st.rerun()


def tab_guidance_lab(model, device, conditioned, gh, gw, num_steps,
                     temperature, base_kwargs, seed):
    st.write("Same seed, same prompt, different guidance — watch the dial "
             "act like an object-count knob.")
    if not conditioned:
        st.info("Needs a text-conditioned checkpoint.")
        return
    prompt = st.text_input("Prompt", value="a cat", key="lab_prompt")
    scales = st.multiselect("Guidance scales", [1.0, 1.5, 2.0, 3.0, 5.0],
                            default=[1.5, 2.0, 3.0])
    if st.button("▷ RUN SWEEP", type="primary") and prompt.strip() and scales:
        try:
            toks, msk = embed_prompt(prompt.strip())
        except ImportError as e:
            st.warning(str(e))
            return
        sweeps = []
        with st.spinner("Sweeping..."):
            for g in sorted(scales):
                torch.manual_seed(seed)
                _, final = generate(model, gh, gw, num_steps=num_steps,
                                    temperature=temperature, device=device,
                                    cond_tokens=toks, cond_mask=msk,
                                    guidance_scale=g,
                                    **{k: v for k, v in base_kwargs.items()
                                       if k not in ("cond_tokens",
                                                    "cond_mask",
                                                    "guidance_scale")})
                sweeps.append((g, final))
        st.session_state["lab_results"] = (prompt, sweeps)

    if "lab_results" in st.session_state:
        prompt, sweeps = st.session_state["lab_results"]
        st.caption(f'"{prompt}"')
        cols = st.columns(len(sweeps))
        for col, (g, final) in zip(cols, sweeps):
            ink = int((final > 2).sum())
            col.markdown(f"**guidance {g:g}** · ink {ink}")
            col.code(grid_to_string(final), language=None)


def tab_training_progress():
    """Browse the per-epoch samples written by training."""
    sample_dir = os.path.join(os.path.dirname(__file__), "..",
                              "checkpoints", "samples")
    files = sorted(glob.glob(os.path.join(sample_dir, "*_epoch*.txt")))
    if not files:
        st.info("No training samples found. They are written to "
                "`checkpoints/samples/` during training — one generated "
                "sample per epoch.")
        return

    by_stage = {}
    for path in files:
        stage, _, epoch_part = os.path.basename(path)[:-4].rpartition("_epoch")
        try:
            epoch = int(epoch_part)
        except ValueError:
            continue
        by_stage.setdefault(stage, []).append((epoch, path))

    stages = sorted(by_stage, key=lambda s: (STAGE_ORDER.get(s, 99), s))
    stage = st.selectbox("Stage", stages)
    epochs = sorted(by_stage[stage])
    if len(epochs) > 1:
        idx = st.slider("Epoch", 1, len(epochs), len(epochs),
                        help="Drag to scrub through the stage's epochs.")
    else:
        idx = 1
    epoch, path = epochs[idx - 1]
    st.caption(f"Stage **{stage}**, epoch {epoch} "
               f"({len(epochs)} epochs recorded)")
    with open(path) as f:
        st.code(f.read(), language=None)

    if len(epochs) > 1 and st.button("▷ play all epochs"):
        ph = st.empty()
        cap = st.empty()
        for e, p in epochs:
            with open(p) as f:
                ph.code(f.read(), language=None)
            cap.caption(f"epoch {e}/{epochs[-1][0]}")
            time.sleep(0.6)


def main():
    st.set_page_config(page_title="ASCII Art Transformer", layout="wide",
                       page_icon="▚")
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
    st.code(BANNER, language=None)
    st.caption("MaskGIT-style iterative generation · prompt-conditioned "
               "via CLIP cross-attention")

    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        st.error("No model checkpoint found in `checkpoints/`. "
                 "Train the model first.")
        # Training progress can still be useful mid-run
        tab_training_progress()
        return

    # --- Sidebar --------------------------------------------------------
    st.sidebar.code(" MODEL ", language=None)
    device_choice = st.sidebar.radio(
        "Device", ["auto", "cpu", "cuda"], horizontal=True,
        help="auto picks cuda only when >3GB VRAM is free, so it won't "
             "fight a training run.")
    device_str = pick_device() if device_choice == "auto" else device_choice
    try:
        model, device, conditioned = load_model(ckpt_path, device_str)
    except ValueError as e:
        st.error(str(e))
        return
    n_params = sum(p.numel() for p in model.parameters())
    st.sidebar.success(f"`{os.path.basename(ckpt_path)}`\n\n"
                       f"{n_params / 1e6:.1f}M params · {device.type} · "
                       f"{'text-conditioned' if conditioned else 'unconditional'}")

    st.sidebar.code(" SAMPLING ", language=None)
    num_steps = st.sidebar.slider("Unmasking steps", 1, 20, UNMASK_STEPS)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, TEMPERATURE, 0.1)
    grid_option = st.sidebar.selectbox("Grid size",
                                       ["48×80", "24×40", "64×128"])
    schedule = st.sidebar.selectbox(
        "Unmask schedule", ["cosine", "linear"],
        help="cosine (MaskGIT): few careful commits early, more late.")
    gumbel_scale = st.sidebar.slider(
        "Exploration (Gumbel)", 0.0, 2.0, 1.0, 0.1,
        help="Annealed noise on the commit order. 0 = greedy.")
    revision_steps = st.sidebar.slider(
        "Refinement passes", 0, 5, 2,
        help="Re-mask the least-confident cells and refill.")
    space_bias = st.sidebar.slider(
        "Ink boost (anti-blank)", 0.0, 8.0, 0.0, 0.5,
        help="Pushes the empty canvas to place ink; fades out as the grid "
             "fills. Raise it (2-6) if generations come out blank.")
    seed = st.sidebar.number_input("Seed", 0, 2**31 - 1, 0)
    randomize = st.sidebar.toggle("New seed every run", value=True)

    if grid_option == "24×40":
        gh, gw = 24, 40
    elif grid_option == "64×128":
        gh, gw = 64, 128  # RoPE extrapolates beyond the 48×80 training size
    else:
        gh, gw = GRID_H, GRID_W

    base_kwargs = dict(schedule=schedule, gumbel_scale=gumbel_scale,
                       revision_steps=revision_steps, space_bias=space_bias)

    tabs = st.tabs(["⚡ Generate", "▦ Inpaint", "◫ Guidance lab",
                    "▁▃▅ Training progress"])
    with tabs[0]:
        tab_generate(model, device, conditioned, gh, gw, num_steps,
                     temperature, base_kwargs, int(seed), randomize)
    with tabs[1]:
        tab_inpaint(model, device, conditioned, gh, gw, num_steps,
                    temperature, base_kwargs)
    with tabs[2]:
        tab_guidance_lab(model, device, conditioned, gh, gw, num_steps,
                         temperature, base_kwargs, int(seed))
    with tabs[3]:
        tab_training_progress()


if __name__ == "__main__":
    main()
