# Deploying nASCIIente

The site is one FastAPI process (model + static frontend) and ships as a
Docker image (`Dockerfile` at the repo root). The checkpoint is **not**
baked into the image — the server downloads it from a Hugging Face model
repo on first boot via `ASCII_CHECKPOINT_HF`.

CPU inference budget: ~34.5M params, a 32-step organic decode takes a
few seconds on a modern core (precise mode several times more). One
shared-CPU instance is fine for a personal showcase; it is not built for
concurrent traffic.

## Step 1 — publish the checkpoint (once)

```bash
pip install -U huggingface_hub
hf auth login                       # paste a WRITE token from hf.co/settings/tokens
hf repo create nasciiente-model --repo-type model --private
hf upload <youruser>/nasciiente-model checkpoints/final_model.pt final_model.pt
```

This doubles as the off-site backup for the model.

## Step 2 — host on Hugging Face Spaces (free)

1. Create a Space at hf.co/new-space — name `nasciiente`, SDK **Docker**,
   hardware **CPU basic** (free).
2. Push this repo to the Space (add it as a git remote), or upload the
   files. The Space needs a `README.md` whose front matter declares the
   Docker SDK — add this at the very top of the Space's README:

   ```yaml
   ---
   title: nASCIIente
   emoji: 🌅
   sdk: docker
   app_port: 7860
   ---
   ```

3. In Space settings → Variables and secrets, set:
   - `ASCII_CHECKPOINT_HF` = `<youruser>/nasciiente-model`
   - `HF_TOKEN` = a READ token (secret; needed because the model repo is
     private — skip if you made it public)
4. First boot downloads the checkpoint plus the frozen CLIP text encoder
   (~600 MB total, cached afterwards). The Space then serves the full
   site at `https://<youruser>-nasciiente.hf.space`.

Free Spaces sleep after ~48h without visitors and wake on the next
request (cold start ≈ a minute).

## Step 3 — the domain

Register `nasciiente.com` / `.art` / `.dev` (~$10–15/yr) at a registrar
that sells at cost (Cloudflare Registrar, Porkbun). Two ways to use it:

- **Redirect (free, works with Spaces):** put the domain on Cloudflare
  DNS and add a Redirect Rule sending `nasciiente.art/*` to the Space
  URL. Spaces do not support custom domains directly, so the URL in the
  browser becomes the Space's — fine for a showcase link.
- **Real custom domain (needs your own host, ~$5/mo):** deploy the same
  Dockerfile to Fly.io (`fly launch` detects it; give the VM 2 GB RAM;
  set the two env vars with `fly secrets set`), then `fly certs add
  nasciiente.art` and point the DNS records it prints. The domain then
  serves the site directly, TLS included. Any small VPS behind Caddy
  works identically.

Start with the Space + redirect; move to Fly/VPS only if the sleep-wake
or the redirect bothers you.

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
