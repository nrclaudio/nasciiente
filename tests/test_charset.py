import torch

from data.charset import (
    PAD_TOKEN, MASK_TOKEN, char_to_idx, idx_to_char,
    grid_to_string, string_to_grid,
)


def test_special_tokens_distinct():
    assert PAD_TOKEN == 0
    assert MASK_TOKEN == 1
    # Printable chars start at index 2 — no collision with special tokens
    assert char_to_idx(" ") == 2
    assert char_to_idx("~") == 96  # 95 printable chars -> indices 2..96


def test_char_round_trip():
    for c in range(32, 127):
        ch = chr(c)
        assert idx_to_char(char_to_idx(ch)) == ch


def test_special_token_rendering():
    assert idx_to_char(PAD_TOKEN) == " "
    assert idx_to_char(MASK_TOKEN) == "?"


def test_string_grid_round_trip():
    text = "+--+\n|ab|\n+--+"
    grid = string_to_grid(text)
    assert grid.shape == (3, 4)
    assert grid_to_string(grid) == text


def test_string_to_grid_pads_short_lines():
    grid = string_to_grid("ab\nc")
    assert grid.shape == (2, 2)
    # Short line padded with spaces
    assert grid[1, 1].item() == char_to_idx(" ")


def test_grid_to_string_shape():
    grid = torch.full((5, 10), char_to_idx("#"), dtype=torch.long)
    s = grid_to_string(grid)
    lines = s.split("\n")
    assert len(lines) == 5
    assert all(line == "#" * 10 for line in lines)
