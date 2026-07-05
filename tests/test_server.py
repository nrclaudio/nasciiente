import os

import torch
import pytest

from model.ascii_bert import ASCIIBert


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    ckpt = tmp_path_factory.mktemp("ck") / "final_model.pt"
    torch.save({"model_state_dict": ASCIIBert().state_dict()}, ckpt)
    os.environ["ASCII_CHECKPOINT"] = str(ckpt)
    os.environ["ASCII_DEVICE"] = "cpu"
    from app import server
    server._STATE.clear()
    return TestClient(server.app)


def test_info(client):
    info = client.get("/api/info").json()
    assert info["conditioned"] is True
    assert info["params_m"] > 30
    assert info["device"] == "cpu"


def test_generate_unconditional(client):
    resp = client.post("/api/generate", json={
        "rows": 12, "cols": 20, "steps": 2, "variations": 2,
        "revision_steps": 0, "seed": 7})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2
    r = data["results"][0]
    assert len(r["final"].split("\n")) == 12
    assert len(r["steps"]) >= 2          # animation frames present
    assert r["seed"] == 7
    assert data["prompt_used"] is False


def test_generate_validates(client):
    resp = client.post("/api/generate", json={"rows": 9999})
    assert resp.status_code == 422       # pydantic bounds

    resp = client.post("/api/generate", json={"variations": 4,
                                              "steps": 99})
    assert resp.status_code == 422


def test_progress_and_static(client):
    assert "samples" in client.get("/api/progress").json()
    page = client.get("/")
    assert page.status_code == 200
    assert "GENERATE" in page.text
