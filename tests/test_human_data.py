import torch

from config import GRID_H, GRID_W
from data.charset import char_to_idx, grid_to_string
from data.prepare_human_ascii import text_to_training_grid, extract_text

SPACE = char_to_idx(" ")

CAT = r"""
 /\_/\
( o.o )
 > ^ <
""" * 3  # repeat to clear the min-ink threshold


def test_valid_art_is_centered():
    grid = text_to_training_grid(CAT)
    assert grid is not None
    assert grid.shape == (GRID_H, GRID_W)
    # Content is centered: first and last rows/cols stay blank
    assert (grid[0] == SPACE).all() and (grid[-1] == SPACE).all()
    assert (grid[:, 0] == SPACE).all() and (grid[:, -1] == SPACE).all()
    # Round trip preserves the art (modulo centering offset)
    assert "( o.o )" in grid_to_string(grid)


def test_rejects_too_wide():
    art = ("#" * (GRID_W + 1) + "\n") * 3
    assert text_to_training_grid(art) is None


def test_rejects_too_tall():
    art = "##########\n" * (GRID_H + 1)
    assert text_to_training_grid(art) is None


def test_rejects_non_ascii():
    art = "┌──────────┐\n│ nice box │\n└──────────┘\n" * 3
    assert text_to_training_grid(art) is None


def test_rejects_too_sparse():
    assert text_to_training_grid(".\n.\n") is None


def test_strips_blank_lines_and_tabs():
    art = "\n\n\t###  ###\n\t###  ###\n\t###  ###\n\n\n"
    grid = text_to_training_grid(art, min_ink=10)
    assert grid is not None
    lines = [l for l in grid_to_string(grid).split("\n") if l.strip()]
    assert len(lines) == 3  # blank lines trimmed
    assert all("\t" not in l for l in lines)


def test_extract_text_finds_multiline_field():
    assert extract_text({"content": "a\nb", "id": 3}) == "a\nb"
    assert extract_text({"caption": "one line", "art": "x\ny"}) == "x\ny"
    assert extract_text({"caption": "one line only"}) is None
