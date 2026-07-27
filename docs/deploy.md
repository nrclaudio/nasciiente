# Deploying nASCIIente

> **Current choice: no always-on server.** The public face of the
> project is the white paper on GitHub Pages
> (`.github/workflows/pages.yml` publishes `docs/paper/` on every push
> to main; the custom domain points there), the weights are public on
> the HF model repo, and the live demo runs locally when needed. The
> options below remain the upgrade path if live public generation ever
> becomes worth hosting.

The site is one FastAPI process (model + static frontend) and ships as a
Docker image (`Dockerfile` at the repo root). The checkpoint is **not**
baked into the image — the server downloads it from a Hugging Face model
repo on first boot via `ASCII_CHECKPOINT_HF`.

CPU inference budget: ~34.5M params, a 32-step organic decode takes a
few seconds on a modern core (precise mode several times more). One
shared-CPU instance is fine for a personal showcase; it is not built for
concurrent traffic.

> **Why not Hugging Face Spaces?** As of mid-2026 the Docker Space SDK
> and the free CPU Basic tier are paywalled behind PRO ($9/mo) for new
> Spaces. HF **model repos remain free**, so the checkpoint hosting
> below still uses HF — only the app hosting moved elsewhere. If you
> ever have PRO anyway, the Dockerfile works on a Docker Space
> unchanged (front matter `sdk: docker`, `app_port: 7860`).

## Step 1 — publish the checkpoint (once, free)

```bash
pip install -U huggingface_hub
hf auth login                       # paste a WRITE token from hf.co/settings/tokens
hf repo create nasciiente-model --repo-type model --private
hf upload <youruser>/nasciiente-model checkpoints/final_model.pt final_model.pt
```

This doubles as the off-site backup for the model.

## Step 2 — pick a host

### Option A — small VPS (recommended: ~€4/mo, real domain, always on)

Hetzner CX22 / CPX11 (or any 2 GB VPS). Once, on the fresh server:

```bash
apt-get update && apt-get install -y docker.io caddy git
git clone https://github.com/<you>/ascii-art-transformer && cd ascii-art-transformer
docker build -t nasciiente .
docker run -d --restart unless-stopped -p 127.0.0.1:7860:7860 \
  -e ASCII_CHECKPOINT_HF=<youruser>/nasciiente-model \
  -e HF_TOKEN=hf_... nasciiente
```

Then point Caddy at it — `/etc/caddy/Caddyfile`:

```
nasciiente.art {
    reverse_proxy localhost:7860
}
```

`systemctl reload caddy` and Caddy provisions TLS automatically once the
domain's DNS A record points at the server. That's the whole stack.

### Option B — Google Cloud Run (free at showcase traffic, scales to zero)

Cloud Run's free tier comfortably covers a personal demo; you pay
nothing while nobody visits. Needs a GCP account (card on file).

```bash
gcloud run deploy nasciiente --source . --region europe-west1 \
  --memory 2Gi --cpu 2 --allow-unauthenticated \
  --set-env-vars ASCII_CHECKPOINT_HF=<youruser>/nasciiente-model \
  --set-secrets HF_TOKEN=hf-token:latest
```

Trade-off: after idle periods the first request cold-starts the
container and re-downloads the checkpoint + CLIP encoder (~a minute).
Mitigate by baking the checkpoint into the image (`COPY` it and drop
the env var) so cold start is image-pull only. Custom domain: Cloud
Run domain mapping, or front it with Cloudflare.

### Option C — your own Mac + Cloudflare Tunnel ($0, instant)

Zero hosting cost, real domain, TLS — the server is just your Mac, so
the site is up only while the Mac is awake:

```bash
brew install cloudflared
cloudflared tunnel login
cloudflared tunnel create nasciiente
cloudflared tunnel route dns nasciiente nasciiente.art
cloudflared tunnel run --url http://localhost:8081 nasciiente
```

with the server running locally (`uvicorn app.server:app --port 8081`
— on Apple Silicon it uses MPS and is faster than any cheap VPS).
Good for showing people this week while deciding on A/B.

## Step 3 — the domain

Register `nasciiente.com` / `.art` / `.dev` (~$10–15/yr) at a registrar
that sells at cost (Cloudflare Registrar, Porkbun). DNS then depends on
the host: an A record to the VPS (Caddy handles TLS), Cloud Run's
domain-mapping records, or the Tunnel's CNAME (created automatically by
`tunnel route dns`).

## Local / self-hosted

```bash
docker build -t nasciiente .
docker run -p 8081:7860 \
  -e ASCII_CHECKPOINT_HF=<youruser>/nasciiente-model \
  -e HF_TOKEN=hf_... nasciiente
```

or mount a local checkpoint instead of downloading:

```bash
docker run -p 8081:7860 \
  -v $PWD/checkpoints/final_model.pt:/srv/nasciiente/checkpoints/final_model.pt \
  nasciiente
```
