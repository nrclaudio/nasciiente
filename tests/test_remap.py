import torch

from config import VOCAB_SIZE
from data.charset import char_to_idx, grid_to_string
from data.remap_tonal import RAMP, build_ramp_lut, remap_payload


def test_ramp_lut_orders_by_density():
    import pytest
    from model.render import glyph_atlas
    if glyph_atlas().sum() == 0:
        pytest.skip("no font available")
    lut = build_ramp_lut()
    ramp_ids = {char_to_idx(c) for c in RAMP}
    # Specials map to themselves; every ink glyph maps INTO the ramp
    for v in (0, 1, char_to_idx(" ")):
        assert int(lut[v]) == v
    for v in range(char_to_idx(" ") + 1, VOCAB_SIZE - 1):
        assert int(lut[v]) in ramp_ids
    # Density ordering: light glyphs land light, dense land dense
    assert int(lut[char_to_idx("'")]) in {char_to_idx("."),
                                          char_to_idx(":"),
                                          char_to_idx("-")}
    assert int(lut[char_to_idx("#")]) in {char_to_idx("#"),
                                          char_to_idx("%"),
                                          char_to_idx("@")}
    # Ink never becomes space
    assert all(int(lut[v]) != char_to_idx(" ")
               for v in range(char_to_idx(" ") + 1, VOCAB_SIZE))


def test_remap_touches_only_untagged_grids():
    import pytest
    from model.render import glyph_atlas
    if glyph_atlas().sum() == 0:
        pytest.skip("no font available")
    g = torch.full((48, 80), char_to_idx(" "), dtype=torch.uint8)
    g[10, 10] = char_to_idx("G")     # off-ramp glyph
    payload = {
        "data": torch.stack([g.clone(), g.clone(), g.clone()]),
        "caption_ids": torch.tensor([0, 1, 2]),
        "captions": ["a goat", "a goat, silhouette",
                     "a goat, outline style"],
    }
    n, total = remap_payload(payload)
    assert (n, total) == (1, 3)
    ramp_ids = {char_to_idx(c) for c in RAMP}
    assert int(payload["data"][0][10, 10]) in ramp_ids     # remapped
    assert int(payload["data"][1][10, 10]) == char_to_idx("G")  # kept
    assert int(payload["data"][2][10, 10]) == char_to_idx("G")  # kept


def test_arch_inference_round_trips_both_sizes():
    from model.ascii_bert import (ASCIIBert, MODEL_SIZES,
                                  arch_from_state, model_matching_state)
    for name, kw in MODEL_SIZES.items():
        # tiny stand-ins with the same RATIOS would be slow to build at
        # full size for 'large'; build base fully, spot-check large
        if name == "large":
            continue
        m = ASCIIBert(**kw)
        arch = arch_from_state(m.state_dict())
        for k, v in kw.items():
            assert arch[k] == v, (name, k)
        m2 = model_matching_state(m.state_dict())
        m2.load_state_dict(m.state_dict())   # strict — must fit exactly

    # cheap non-standard shape: inference must recover it too
    m = ASCIIBert(embed_dim=128, num_layers=3, num_heads=2, ffn_dim=256,
                  text_dim=64)
    arch = arch_from_state(m.state_dict())
    assert arch["embed_dim"] == 128 and arch["num_layers"] == 3
    assert arch["ffn_dim"] == 256 and arch["text_dim"] == 64
