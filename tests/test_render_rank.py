import torch

from config import VOCAB_SIZE
from data.charset import char_to_idx
from model.render import glyph_atlas, render_grid, render_probs, \
    render_to_pil


def test_glyph_atlas_shape_and_content():
    atlas = glyph_atlas()
    assert atlas.shape[0] == VOCAB_SIZE
    assert atlas.min() >= 0 and atlas.max() <= 1
    # PAD/MASK/space blank; a dense glyph has ink
    assert atlas[0].sum() == 0 and atlas[1].sum() == 0
    assert atlas[char_to_idx(" ")].sum() == 0
    assert atlas[char_to_idx("@")].sum() > 0


def test_render_grid_blank_and_ink():
    space = char_to_idx(" ")
    grid = torch.full((4, 6), space, dtype=torch.long)
    img = render_grid(grid)
    assert img.shape == (4 * 12, 6 * 8)
    assert img.sum() == 0
    grid[2, 3] = char_to_idx("@")
    img = render_grid(grid)
    # Ink lands exactly in that cell's tile
    assert img[2 * 12:3 * 12, 3 * 8:4 * 8].sum() > 0
    assert img.sum() == img[2 * 12:3 * 12, 3 * 8:4 * 8].sum()


def test_render_probs_matches_hard_render_and_is_differentiable():
    torch.manual_seed(0)
    grid = torch.randint(2, VOCAB_SIZE, (3, 5))
    onehot = torch.zeros(3, 5, VOCAB_SIZE)
    onehot.scatter_(-1, grid.unsqueeze(-1), 1.0)
    assert torch.allclose(render_probs(onehot), render_grid(grid),
                          atol=1e-6)
    # Gradients flow through the soft render (the future training hook)
    probs = torch.softmax(torch.randn(3, 5, VOCAB_SIZE,
                                      requires_grad=True), dim=-1)
    render_probs(probs).sum().backward()


def test_render_to_pil():
    grid = torch.full((4, 6), char_to_idx("#"), dtype=torch.long)
    img = render_to_pil(grid)
    assert img.mode == "RGB"
    assert img.size == (6 * 8, 4 * 12)  # PIL size is (W, H)


def test_clip_scores_plumbing(monkeypatch):
    import data.clip_rank as CR

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeOut:
        def __init__(self, n):
            e = torch.eye(4)[:n]
            self.image_embeds = e            # orthonormal rows
            self.text_embeds = e[1:2]        # matches image 1 exactly

    class FakeModel:
        def __call__(self, **inputs):
            return FakeOut(inputs["n"])

    class FakeProcessor:
        def __call__(self, text=None, images=None, return_tensors=None,
                     padding=None, truncation=None):
            return FakeInputs(n=len(images))

    monkeypatch.setattr(CR, "_load_scorer",
                        lambda device="cpu": (FakeModel(), FakeProcessor(),
                                              "cpu"))
    grids = [torch.full((2, 3), 2, dtype=torch.long) for _ in range(3)]
    scores = CR.clip_scores(grids, "a cat")
    assert scores.shape == (3,)
    assert scores.argmax().item() == 1  # the render CLIP "recognized"


def test_auto_caption_grids_cleaning():
    from data.prepare_human_ascii import _clean_caption, auto_caption_grids
    assert _clean_caption("A drawing of a cat") == "a cat"
    assert _clean_caption(" ascii art of  a DOG ") == "a DOG"
    assert _clean_caption("a sketch of a boat on water") == "a boat on water"
    assert _clean_caption("castle") == "castle"
    assert _clean_caption("   ") is None

    grids = [torch.full((4, 6), 2, dtype=torch.long) for _ in range(5)]
    calls = []

    def fake_captioner(images):
        calls.append(len(images))
        return [f"a drawing of thing {len(calls)}-{i}"
                for i in range(len(images))]

    captions = auto_caption_grids(grids, batch_size=2,
                                  captioner=fake_captioner)
    assert len(captions) == 5
    assert calls == [2, 2, 1]  # batched
    assert captions[0] == "thing 1-0"
