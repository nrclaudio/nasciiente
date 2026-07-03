"""Generate synthetic ASCII geometry data.

Produces [N, 48, 80] long tensors containing randomized geometric primitives.
"""

import os
import sys
import random

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRID_H, GRID_W, GEOMETRY_NUM_SAMPLES
from data.charset import char_to_idx, grid_to_string

SPACE = char_to_idx(" ")


def _blank_grid():
    return torch.full((GRID_H, GRID_W), SPACE, dtype=torch.long)


def _clamp(r, c):
    return max(0, min(r, GRID_H - 1)), max(0, min(c, GRID_W - 1))


def _set(grid, r, c, ch_idx):
    if 0 <= r < GRID_H and 0 <= c < GRID_W:
        grid[r, c] = ch_idx


def draw_hline(grid):
    r = random.randint(0, GRID_H - 1)
    c0 = random.randint(0, GRID_W - 10)
    length = random.randint(5, min(40, GRID_W - c0))
    ch = char_to_idx("-")
    for c in range(c0, c0 + length):
        _set(grid, r, c, ch)


def draw_vline(grid):
    c = random.randint(0, GRID_W - 1)
    r0 = random.randint(0, GRID_H - 5)
    length = random.randint(3, min(20, GRID_H - r0))
    ch = char_to_idx("|")
    for r in range(r0, r0 + length):
        _set(grid, r, c, ch)


def draw_diagonal(grid):
    direction = random.choice(["/", "\\"])
    ch = char_to_idx(direction)
    r0 = random.randint(2, GRID_H - 3)
    c0 = random.randint(2, GRID_W - 3)
    length = random.randint(3, 15)
    for i in range(length):
        if direction == "\\":
            _set(grid, r0 + i, c0 + i, ch)
        else:
            _set(grid, r0 - i, c0 + i, ch)


def draw_rectangle(grid):
    r0 = random.randint(0, GRID_H - 6)
    c0 = random.randint(0, GRID_W - 10)
    h = random.randint(4, min(20, GRID_H - r0))
    w = random.randint(6, min(30, GRID_W - c0))
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    dash = char_to_idx("-")
    pipe = char_to_idx("|")
    plus = char_to_idx("+")
    # corners
    for rr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
        _set(grid, rr, cc, plus)
    # top and bottom edges
    for c in range(c0 + 1, c1):
        _set(grid, r0, c, dash)
        _set(grid, r1, c, dash)
    # left and right edges
    for r in range(r0 + 1, r1):
        _set(grid, r, c0, pipe)
        _set(grid, r, c1, pipe)


def draw_triangle(grid):
    """Right triangle growing down-right."""
    r0 = random.randint(0, GRID_H - 8)
    c0 = random.randint(0, GRID_W - 15)
    size = random.randint(4, min(12, GRID_H - r0, GRID_W - c0))
    slash = char_to_idx("/")
    dash = char_to_idx("-")
    pipe = char_to_idx("|")
    plus = char_to_idx("+")
    # vertical left edge
    for r in range(r0, r0 + size):
        _set(grid, r, c0, pipe)
    # horizontal bottom edge
    for c in range(c0, c0 + size):
        _set(grid, r0 + size - 1, c, dash)
    # diagonal hypotenuse
    for i in range(size):
        _set(grid, r0 + size - 1 - i, c0 + i, slash)
    # corner
    _set(grid, r0 + size - 1, c0, plus)


def draw_cross(grid):
    cr = random.randint(5, GRID_H - 6)
    cc = random.randint(5, GRID_W - 6)
    arm = random.randint(3, min(8, cr, GRID_H - 1 - cr, cc, GRID_W - 1 - cc))
    dash = char_to_idx("-")
    pipe = char_to_idx("|")
    plus = char_to_idx("+")
    _set(grid, cr, cc, plus)
    for i in range(1, arm + 1):
        _set(grid, cr - i, cc, pipe)
        _set(grid, cr + i, cc, pipe)
        _set(grid, cr, cc - i, dash)
        _set(grid, cr, cc + i, dash)


def draw_diamond(grid):
    cr = random.randint(5, GRID_H - 6)
    cc = random.randint(8, GRID_W - 9)
    size = random.randint(3, min(6, cr, GRID_H - 1 - cr, cc // 2, (GRID_W - 1 - cc) // 2))
    slash = char_to_idx("/")
    bslash = char_to_idx("\\")
    dash = char_to_idx("-")
    # top half
    for i in range(size):
        _set(grid, cr - size + i, cc - i - 1, slash)
        _set(grid, cr - size + i, cc + i + 1, bslash)
    # bottom half
    for i in range(size):
        _set(grid, cr + i + 1, cc - size + i + 1, bslash)
        _set(grid, cr + i + 1, cc + size - i - 1, slash)
    # horizontal tips
    _set(grid, cr, cc - size, dash)
    _set(grid, cr, cc + size, dash)


def draw_text_label(grid):
    labels = ["Hello", "ASCII", "Box", "Test", "Data", "Grid", "Art", "Cool", "Wow", "OK"]
    text = random.choice(labels)
    r = random.randint(0, GRID_H - 1)
    c = random.randint(0, GRID_W - len(text))
    for i, ch in enumerate(text):
        _set(grid, r, c + i, char_to_idx(ch))


PRIMITIVES = [
    draw_hline,
    draw_vline,
    draw_diagonal,
    draw_rectangle,
    draw_triangle,
    draw_cross,
    draw_diamond,
    draw_text_label,
]


def generate_sample():
    grid = _blank_grid()
    n_prims = random.randint(1, 4)
    for _ in range(n_prims):
        fn = random.choice(PRIMITIVES)
        fn(grid)
    return grid


def main():
    print(f"Generating {GEOMETRY_NUM_SAMPLES} geometry samples ({GRID_H}x{GRID_W})...")

    # Pre-allocate to avoid OOM from a huge Python list.
    # uint8 (vocab is 98 < 256) keeps 200k grids at ~0.8 GB, not 6 GB.
    data = torch.full((GEOMETRY_NUM_SAMPLES, GRID_H, GRID_W), SPACE, dtype=torch.uint8)
    for i in range(GEOMETRY_NUM_SAMPLES):
        data[i] = generate_sample()
        if (i + 1) % 50_000 == 0:
            print(f"  {i + 1}/{GEOMETRY_NUM_SAMPLES}")

    print(f"Tensor shape: {data.shape}")

    save_dir = os.path.join(os.path.dirname(__file__))
    save_path = os.path.join(save_dir, "geometry_data.pt")
    torch.save(data, save_path)
    print(f"Saved to {save_path}")

    # Print sample grids for visual verification
    print("\n=== Sample Geometry Grids ===\n")
    for i in range(5):
        print(f"--- Sample {i} ---")
        print(grid_to_string(data[i]))
        print()


if __name__ == "__main__":
    main()
