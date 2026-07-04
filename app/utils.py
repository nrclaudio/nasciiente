import torch

from data.charset import MASK_TOKEN, char_to_idx, string_to_grid


def text_to_partial_grid(text, grid_h, grid_w):
    """
    Convert user text input to a partial grid for inpainting.

    Spaces remain as spaces (not masked). Empty lines / short lines
    are padded with MASK tokens.

    Args:
        text: multi-line string from user
        grid_h: target grid height
        grid_w: target grid width

    Returns:
        [grid_h, grid_w] long tensor with MASK_TOKEN for empty positions
    """
    grid = torch.full((grid_h, grid_w), MASK_TOKEN, dtype=torch.long)
    lines = text.split("\n")

    for r, line in enumerate(lines):
        if r >= grid_h:
            break
        for c, ch in enumerate(line):
            if c >= grid_w:
                break
            try:
                grid[r, c] = char_to_idx(ch)
            except KeyError:
                grid[r, c] = MASK_TOKEN

    return grid


def count_fixed_positions(grid):
    """Count non-MASK positions in a grid."""
    return (grid != MASK_TOKEN).sum().item()


# Terminal-ish color schemes for PNG export: (background, foreground)
PNG_SCHEMES = {
    "green": ((6, 12, 6), (74, 246, 38)),
    "amber": ((16, 10, 2), (255, 176, 0)),
    "paper": ((248, 246, 240), (24, 24, 24)),
}


def grid_to_png_bytes(grid, scheme="green", cell_w=9, cell_h=16, pad=24):
    """Render a character grid to PNG bytes (for download/sharing).

    Draws each cell with a monospace font on a terminal-style background.
    """
    import io

    from PIL import Image, ImageDraw

    from data.charset import idx_to_char
    from data.glyph_sim import _find_mono_font

    bg, fg = PNG_SCHEMES.get(scheme, PNG_SCHEMES["green"])
    h, w = grid.shape
    img = Image.new("RGB", (w * cell_w + 2 * pad, h * cell_h + 2 * pad), bg)
    draw = ImageDraw.Draw(img)
    font = _find_mono_font(14)
    for r in range(h):
        line = "".join(idx_to_char(int(grid[r, c])) for c in range(w))
        draw.text((pad, pad + r * cell_h), line, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
