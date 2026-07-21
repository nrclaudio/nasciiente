# nASCIIente serving image — works as-is on Hugging Face Spaces
# (Docker SDK, app_port 7860), Fly.io, or any container host.
#
# The checkpoint is NOT baked in: set ASCII_CHECKPOINT_HF to a HF
# model repo (e.g. "youruser/nasciiente-model") and the server
# downloads it on first boot. See docs/deploy.md.
FROM python:3.11-slim

# The glyph atlas and image converter rasterize through a real
# monospace font; without one they degrade to blank bitmaps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/nasciiente

COPY requirements-serve.txt .
RUN pip install --no-cache-dir torch \
      --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY config.py ./
COPY app/ app/
COPY model/ model/
COPY data/ data/

# Spaces run as a non-root user with $HOME=/home/user; the HF cache
# (checkpoint + CLIP text encoder) must land somewhere writable
ENV ASCII_DEVICE=cpu \
    HF_HOME=/tmp/hf-cache

EXPOSE 7860
CMD ["sh", "-c", "python -m uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
