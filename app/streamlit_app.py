"""
Streamlit web interface for ASCII Art Transformer.

Modes:
  - Generate from scratch (fully masked → iterative unmasking)
  - Inpainting (user provides partial ASCII, model fills the rest)
"""

import os
import sys
import time

import streamlit as st
import torch

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W, UNMASK_STEPS, TEMPERATURE, VOCAB_SIZE
from model.ascii_bert import ASCIIBert
from model.inference import generate, upscale_grid
from data.charset import grid_to_string, MASK_TOKEN
from app.utils import text_to_partial_grid, count_fixed_positions


@st.cache_resource
def load_model(checkpoint_path):
    """Load the trained model on CPU (for deployment)."""
    device = torch.device("cpu")
    model = ASCIIBert().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, device


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


def main():
    st.set_page_config(page_title="ASCII Art Transformer", layout="wide")
    st.title("ASCII Art Transformer")
    st.caption("MaskGIT-style iterative generation of ASCII art")

    # Load model
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        st.error("No model checkpoint found in `checkpoints/`. Train the model first.")
        return

    model, device = load_model(ckpt_path)
    st.sidebar.success(f"Model loaded from `{os.path.basename(ckpt_path)}`")

    # Controls
    st.sidebar.header("Controls")
    mode = st.sidebar.radio("Mode", ["Generate from scratch", "Inpainting"])
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, TEMPERATURE, 0.1)
    num_steps = st.sidebar.slider("Unmasking steps", 1, 20, UNMASK_STEPS)
    grid_option = st.sidebar.selectbox("Grid size", ["48×80", "24×40", "64×128"])

    if grid_option == "24×40":
        gh, gw = 24, 40
    elif grid_option == "64×128":
        gh, gw = 64, 128  # RoPE extrapolates beyond the 48×80 training size
    else:
        gh, gw = GRID_H, GRID_W

    show_animation = st.sidebar.checkbox("Show step-by-step", value=False)

    if mode == "Generate from scratch":
        if st.button("Generate"):
            with st.spinner("Generating..."):
                t0 = time.time()
                steps, final = generate(model, gh, gw,
                                        num_steps=num_steps,
                                        temperature=temperature,
                                        device=device)
                elapsed = time.time() - t0

            st.session_state["last_grid"] = final
            st.session_state["last_elapsed"] = elapsed
            st.session_state["last_steps"] = steps
            st.session_state.pop("upscaled_grid", None)

        if "last_grid" in st.session_state:
            final = st.session_state["last_grid"]
            st.code(grid_to_string(final), language=None)
            st.caption(f"Generated in {st.session_state['last_elapsed']:.1f}s "
                       f"({num_steps} steps)")

            h, w = final.shape
            if st.button(f"Upscale ×2 → {h*2}×{w*2}",
                         help="Anchor each character on a 2× canvas and "
                              "inpaint the gaps (MaskGIT super-resolution)"):
                with st.spinner("Upscaling..."):
                    t0 = time.time()
                    _, upscaled = upscale_grid(model, final, factor=2,
                                               num_steps=num_steps,
                                               temperature=temperature,
                                               device=device)
                st.session_state["upscaled_grid"] = upscaled
                st.session_state["upscale_elapsed"] = time.time() - t0

            if "upscaled_grid" in st.session_state:
                st.subheader("Upscaled ×2")
                st.code(grid_to_string(st.session_state["upscaled_grid"]),
                        language=None)
                st.caption(f"Upscaled in "
                           f"{st.session_state['upscale_elapsed']:.1f}s")

            steps = st.session_state["last_steps"]
            if show_animation and len(steps) > 1:
                st.subheader("Step-by-step")
                for i, step_grid in enumerate(steps):
                    with st.expander(f"Step {i}"):
                        st.code(grid_to_string(step_grid), language=None)

    else:  # Inpainting
        st.write("Enter partial ASCII art below. Empty areas will be filled by the model.")
        user_text = st.text_area(
            "Partial ASCII art",
            height=300,
            placeholder="Type some ASCII characters...\n"
                        "Leave areas blank for the model to fill in.",
        )

        if st.button("Fill in") and user_text.strip():
            partial = text_to_partial_grid(user_text, gh, gw)
            fixed = count_fixed_positions(partial)
            st.caption(f"Fixed positions: {fixed}/{gh*gw}")

            with st.spinner("Inpainting..."):
                t0 = time.time()
                steps, final = generate(model, gh, gw,
                                        num_steps=num_steps,
                                        temperature=temperature,
                                        initial_grid=partial,
                                        device=device)
                elapsed = time.time() - t0

            st.code(grid_to_string(final), language=None)
            st.caption(f"Completed in {elapsed:.1f}s")

            if show_animation and len(steps) > 1:
                st.subheader("Step-by-step")
                for i, step_grid in enumerate(steps):
                    with st.expander(f"Step {i}"):
                        st.code(grid_to_string(step_grid), language=None)


if __name__ == "__main__":
    main()
