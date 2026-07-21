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


def test_cloud_gif(client):
    resp = client.post("/api/cloud", json={
        "rows": 12, "cols": 20, "steps": 3, "variations": 1,
        "revision_steps": 0, "seed": 7})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/gif"
    assert resp.headers["x-seed"] == "7"
    assert resp.content[:6] in (b"GIF87a", b"GIF89a")
    # deterministic: same seed re-renders the identical cloud
    again = client.post("/api/cloud", json={
        "rows": 12, "cols": 20, "steps": 3, "variations": 1,
        "revision_steps": 0, "seed": 7})
    assert again.content == resp.content


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
    assert "GLYPH48" in page.text


def _png_bytes():
    import io
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 60, 200, 200], outline="black", width=8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _gif_bytes(n_frames):
    import io
    from PIL import Image, ImageDraw
    frames = []
    for i in range(n_frames):
        img = Image.new("RGB", (128, 128), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([20 + i * 10, 20, 90 + i * 10, 90],
                       outline="black", width=6)
        frames.append(img)
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=125, loop=0)
    return buf.getvalue()


def test_convert_single_image(client):
    resp = client.post("/api/convert",
                       files={"file": ("box.png", _png_bytes(),
                                       "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["animated"] is False
    assert len(data["frames"]) == 1
    lines = data["frames"][0].split("\n")
    assert len(lines) == 48 and all(len(l) == 80 for l in lines)
    # The rectangle survived conversion: real ink, not a blank grid
    assert sum(c not in " " for l in lines for c in l) > 30


def test_convert_animated_gif(client):
    resp = client.post("/api/convert?binarize=false",
                       files={"file": ("anim.gif", _gif_bytes(3),
                                       "image/gif")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["animated"] is True
    assert len(data["frames"]) == 3
    assert data["fps"] == 8                # 125ms frame duration
    # Frames differ (the box moves)
    assert data["frames"][0] != data["frames"][1]


def test_convert_rejects_non_image(client):
    resp = client.post("/api/convert",
                       files={"file": ("junk.bin", b"not an image",
                                       "application/octet-stream")})
    assert resp.status_code == 422


def test_gif_export(client):
    frames = ["##  \n  ##", "  ##\n##  "]
    resp = client.post("/api/gif", json={"frames": frames, "fps": 8})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/gif"
    assert resp.content[:6] in (b"GIF87a", b"GIF89a")


def test_gif_validates(client):
    assert client.post("/api/gif",
                       json={"frames": []}).status_code == 422
    assert client.post("/api/gif",
                       json={"frames": ["#"],
                             "fps": 99}).status_code == 422
