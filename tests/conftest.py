import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model.ascii_bert import ASCIIBert


@pytest.fixture(scope="session")
def tiny_model():
    """A small ASCIIBert that runs fast on CPU."""
    torch.manual_seed(0)
    model = ASCIIBert(embed_dim=32, num_layers=2, num_heads=2, ffn_dim=64)
    model.eval()
    return model
