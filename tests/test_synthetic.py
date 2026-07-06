import torch

from data.prompt_bank import build_prompts, _article, _plural


def test_prompt_bank_unique_and_deterministic():
    a = build_prompts(2000, seed=0)
    b = build_prompts(2000, seed=0)
    assert a == b
    assert len(set(a)) == 2000
    assert all(p and "\n" not in p for p in a)
    # The bank mixes attributes, counts and compositions
    assert any(p.startswith(("two ", "three ")) for p in a)
    assert any(" and " in p for p in a)


def test_prompt_bank_survives_pool_exhaustion():
    # Asking for more captions than any category pool holds must
    # terminate (this hung the 200k engine run) and cycle with repeats
    from data.prompt_bank import _singles_pool
    n = len(_singles_pool()) * 3
    prompts = build_prompts(n, seed=1)
    assert len(prompts) == n


def test_prompt_bank_mix_is_singles_heavy():
    # Two-subject prompts fuse into chimeras on few-step t2i models, so
    # pairs must be a small fraction — not the accidental 93% a naive
    # combinatorial pool gives
    prompts = build_prompts(10_000, seed=0)
    pair_frac = sum(" and a" in p or " and an" in p for p in prompts) / 1e4
    count_frac = sum(p.startswith(("two ", "three "))
                     for p in prompts) / 1e4
    assert 0.06 < pair_frac < 0.14
    assert 0.10 < count_frac < 0.20


def test_prompt_grammar_helpers():
    assert _article("owl") == "an owl"
    assert _article("cat") == "a cat"
    assert _plural("fox") == "foxes"
    assert _plural("butterfly") == "butterflies"
    assert _plural("cat") == "cats"


class _FakeResult:
    def __init__(self, images):
        self.images = images


class _FakePipe:
    """Stands in for a diffusers pipeline: draws simple line-art PIL
    images so the real converter has something to chew on."""

    def __call__(self, prompt=None, negative_prompt=None,
                 num_inference_steps=None, guidance_scale=None,
                 generator=None):
        from PIL import Image, ImageDraw
        images = []
        for i, _ in enumerate(prompt):
            img = Image.new("RGB", (256, 256), "white")
            draw = ImageDraw.Draw(img)
            draw.rectangle([40 + i * 3, 40, 200, 200], outline="black",
                           width=6)
            draw.line([40, 40, 200, 200], fill="black", width=6)
            images.append(img)
        return _FakeResult(images)


def test_generate_dataset_end_to_end(tmp_path):
    from data.generate_synthetic import generate_dataset
    from training.train import _load_data_and_captions

    # A merge source in the standard payload format
    merge_src = tmp_path / "old.pt"
    torch.save({"data": torch.randint(2, 98, (5, 48, 80), dtype=torch.uint8),
                "caption_ids": torch.tensor([0, 1, -1, 0, 1]),
                "captions": ["old1", "old2"]}, merge_src)

    out = tmp_path / "synth.pt"
    n = generate_dataset(num_samples=6, out_path=str(out), batch_size=3,
                         seed=0, merge=str(merge_src), device="cpu",
                         pipe=_FakePipe())
    assert n == 11  # 6 generated + 5 merged

    data, ids, caps = _load_data_and_captions(str(out))
    assert data.shape == (11, 48, 80) and data.dtype == torch.uint8
    assert len(ids) == 11
    # Generated grids actually contain ink (the drawn rectangle survived
    # conversion) and their captions come first in the merged table
    assert int((data[0] > 2).sum()) > 50
    assert caps[-2:] == ["old1", "old2"]
    # Merged ids were offset past the generated captions
    n_gen_caps = len(caps) - 2
    tail = ids[6:]
    assert all(i == -1 or i >= n_gen_caps for i in tail.tolist())
    # And every generated id points into the generated captions
    assert all(0 <= i < n_gen_caps for i in ids[:6].tolist())


def test_binarize_finds_natural_threshold():
    from data.generate_synthetic import _binarize
    # Bimodal image: white page (240ish) with 10% dark strokes (30ish).
    # Otsu must split between the modes — ink fraction ~= stroke fraction,
    # NOT a forced quota
    torch.manual_seed(0)
    img = torch.normal(240.0, 5.0, (100, 80)).clamp(0, 255)
    strokes = torch.rand(100, 80) < 0.10
    img[strokes] = torch.normal(30.0, 10.0, (int(strokes.sum()),)).clamp(0, 255)
    out = _binarize(img.to(torch.uint8))
    assert set(torch.unique(out).tolist()) <= {0, 255}
    frac = float((out == 0).float().mean())
    assert 0.08 < frac < 0.12  # matches the true stroke fraction

    # Shaded gradient still collapses to exactly two tones
    gradient = torch.linspace(0, 255, 100 * 80).view(100, 80).to(torch.uint8)
    assert set(torch.unique(_binarize(gradient)).tolist()) <= {0, 255}


def test_trim_removes_border_bands():
    from data.generate_synthetic import images_to_grids, _build_tables
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 12, 255], fill="black")       # left border band
    draw.rectangle([100, 100, 160, 160], outline="black", width=6)
    tables = _build_tables()
    (grid, _), = images_to_grids([img], tables, binarize=True, trim=0.06)
    assert grid is not None
    # The sidebar must be gone: leftmost columns of the IMAGE area blank.
    # (letterboxing pads ~5% of columns; check just inside that)
    assert int((grid[:, 4:10] > 2).sum()) == 0
    # The actual subject (rectangle) survived
    assert int((grid > 2).sum()) > 20


def test_clip_filter_drops_off_prompt_images(tmp_path):
    from data.generate_synthetic import generate_dataset

    scored = []

    def fake_scorer(images, captions):
        # every second image "doesn't match" its caption
        scores = torch.tensor([0.5 if i % 2 == 0 else 0.05
                               for i in range(len(images))])
        scored.append(len(images))
        return scores

    out = tmp_path / "synth.pt"
    n = generate_dataset(num_samples=4, out_path=str(out), batch_size=4,
                         seed=0, device="cpu", pipe=_FakePipe(),
                         clip_filter=0.2, clip_scorer=fake_scorer)
    assert n == 4
    assert scored  # the filter actually ran


def test_parallel_conversion_matches_serial(tmp_path):
    # The worker pool must produce the same dataset as the serial path
    # (same fake images, same converter, same filters) — including under
    # the v2 style-mode mix, where conversion params vary per batch
    from data.generate_synthetic import generate_dataset

    for tag, modes in [("legacy", None),
                       ("mix", {"filled": 0.4, "outline": 0.3,
                                "tonal": 0.3})]:
        serial = tmp_path / f"serial_{tag}.pt"
        parallel = tmp_path / f"parallel_{tag}.pt"
        generate_dataset(num_samples=6, out_path=str(serial), batch_size=3,
                         seed=0, device="cpu", pipe=_FakePipe(), workers=1,
                         modes=modes)
        generate_dataset(num_samples=6, out_path=str(parallel),
                         batch_size=3, seed=0, device="cpu",
                         pipe=_FakePipe(), workers=2, modes=modes)

        a = torch.load(serial, weights_only=True)
        b = torch.load(parallel, weights_only=True)
        assert torch.equal(a["data"], b["data"]), tag
        assert torch.equal(a["caption_ids"], b["caption_ids"]), tag
        assert a["captions"] == b["captions"], tag


def test_outline_mode_hollows_filled_shapes():
    # A solid black square must convert to a boundary ring: ink on the
    # edges, blank interior — the second visual dialect
    from data.generate_synthetic import images_to_grids, _build_tables
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), "white")
    ImageDraw.Draw(img).rectangle([48, 48, 208, 208], fill="black")
    tables = _build_tables()
    (filled, _), = images_to_grids([img], tables, binarize=True,
                                   max_ink=1.0)
    (outlined, _), = images_to_grids([img], tables, binarize=True,
                                     outline=True, max_ink=1.0)
    # The filled square's interior is inked; the outline's interior isn't
    assert int((filled[20:28, 35:45] > 2).sum()) > 60
    assert int((outlined[20:28, 35:45] > 2).sum()) == 0
    # But the outline still has substantial boundary ink
    assert int((outlined > 2).sum()) > 40


def test_tonal_conversion_keeps_gray_levels():
    # Without binarize, a soft-shaded blob must convert through the tonal
    # pipeline into a range of glyph densities, not two tones
    from data.generate_synthetic import images_to_grids, _build_tables
    from PIL import Image, ImageDraw, ImageFilter
    img = Image.new("L", (256, 256), 255)
    draw = ImageDraw.Draw(img)
    for r, tone in [(100, 200), (80, 150), (60, 100), (40, 50), (20, 10)]:
        draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], fill=tone)
    img = img.filter(ImageFilter.GaussianBlur(8)).convert("RGB")
    tables = _build_tables()
    (tonal, _), = images_to_grids([img], tables, binarize=False,
                                  max_ink=1.0)
    (binary, _), = images_to_grids([img], tables, binarize=True,
                                   max_ink=1.0)
    tonal_glyphs = len(set(tonal[tonal > 2].tolist()))
    binary_glyphs = len(set(binary[binary > 2].tolist()))
    assert tonal_glyphs > binary_glyphs
    assert tonal_glyphs >= 8    # a real density ramp, not a silhouette


def test_flatten_bg_rescues_textured_backgrounds():
    # Reproduces the v2 engine failure: near-white paper texture gets
    # amplified by CLAHE into faint glyphs across the whole canvas, so
    # tonal conversions blow past the ink cap. Flattening the background
    # (estimated from the border) must bring ink back down to the
    # subject alone.
    import numpy as np
    from data.generate_synthetic import images_to_grids, _build_tables
    from PIL import Image, ImageDraw, ImageFilter

    rng = np.random.default_rng(0)
    noise = rng.integers(228, 250, (256, 256)).astype(np.float32)  # paper
    yy, xx = np.mgrid[:256, :256].astype(np.float32)
    dist = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2) / 181.0
    vignette = 1.0 - 0.18 * dist ** 2            # darkened corners
    img = Image.fromarray((noise * vignette).astype(np.uint8), "L")
    draw = ImageDraw.Draw(img)
    for r, tone in [(80, 160), (55, 100), (30, 35)]:            # subject
        draw.ellipse([128 - r, 128 - r, 128 + r, 128 + r], fill=tone)
    img = img.filter(ImageFilter.GaussianBlur(4)).convert("RGB")

    tables = _build_tables()
    (_, raw_ink), = images_to_grids([img], tables, max_ink=1.0,
                                    min_ink=0.0)
    (grid, flat_ink), = images_to_grids([img], tables, max_ink=1.0,
                                        min_ink=0.0, flatten_bg=True)
    # Flattening must strip a substantial background film...
    assert raw_ink - flat_ink > 0.15, (raw_ink, flat_ink)
    assert flat_ink < 0.55, flat_ink   # ...leaving mostly the subject
    # And the subject survived, with a real density ramp
    assert len(set(grid[grid > 2].tolist())) >= 6


def test_style_mix_tags_captions(tmp_path):
    from data.generate_synthetic import generate_dataset, parse_mix

    out = tmp_path / "mix.pt"
    generate_dataset(num_samples=12, out_path=str(out), batch_size=3,
                     seed=0, device="cpu", pipe=_FakePipe(), workers=1,
                     modes={"outline": 0.5, "tonal": 0.5})
    caps = torch.load(out, weights_only=True)["captions"]
    assert caps
    assert all(c.endswith((", outline style", ", shaded")) for c in caps)

    mix = parse_mix("filled=1,tonal=3")
    assert abs(mix["filled"] - 0.25) < 1e-9
    assert abs(mix["tonal"] - 0.75) < 1e-9
    try:
        parse_mix("bogus=1")
        assert False, "unknown mode must raise"
    except ValueError:
        pass
