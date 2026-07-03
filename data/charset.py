import torch

# Special tokens
PAD_TOKEN = 0
MASK_TOKEN = 1

# Printable ASCII: space (32) through ~ (126) → indices 2..97
_PRINTABLE_START = 32  # space
_PRINTABLE_END = 126   # ~
_OFFSET = 2

_char_to_idx = {chr(c): c - _PRINTABLE_START + _OFFSET for c in range(_PRINTABLE_START, _PRINTABLE_END + 1)}
_idx_to_char = {v: k for k, v in _char_to_idx.items()}


def char_to_idx(char: str) -> int:
    return _char_to_idx[char]


def idx_to_char(idx: int) -> str:
    if idx == PAD_TOKEN:
        return " "
    if idx == MASK_TOKEN:
        return "?"
    # VOCAB_SIZE (98) is one wider than the mapped range (0..96); a model
    # can emit the unused slot 97. Render any unmapped index as a space
    # rather than crash the display.
    return _idx_to_char.get(idx, " ")


def grid_to_string(int_grid: torch.Tensor) -> str:
    """Convert a [H, W] int tensor to a multi-line ASCII string."""
    lines = []
    for row in int_grid:
        lines.append("".join(idx_to_char(idx.item()) for idx in row))
    return "\n".join(lines)


def string_to_grid(text: str) -> torch.Tensor:
    """Convert a multi-line ASCII string to a [H, W] int tensor."""
    lines = text.split("\n")
    h = len(lines)
    w = max(len(line) for line in lines) if lines else 0
    grid = torch.full((h, w), char_to_idx(" "), dtype=torch.long)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            if ch in _char_to_idx:
                grid[r, c] = _char_to_idx[ch]
    return grid
