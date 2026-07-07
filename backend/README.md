---
title: Fast-DetectGPT
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Fast-DetectGPT backend

Zero-shot AI-text detection via GPT-2 conditional probability curvature
(Bao et al., ICLR 2024). Free HuggingFace Space, Docker SDK, CPU basic tier — $0.

This exists because the project's 17 stylometric signals were empirically shown
**not to generalize** across datasets (leave-one-source-out AUROC 0.524 ≈ chance;
see `../memory/finding_stylometry_ceiling.md`). Fast-DetectGPT has no learned
decision boundary, so it should transfer where the trained classifier did not.

## Files
- `fast_detect.py` — the scorer (the algorithm; everything else wraps it)
- `app.py` — FastAPI service: `POST /score {text} -> {discrepancy, probability}`
- `Dockerfile` — Python 3.11 + torch (local machine is 3.14, which has no torch wheels)
- `requirements.txt`
- `eval_via_api.js` — gatekeeper: scores `eval/combined_dataset.csv` through the
  deployed endpoint and reports per-source AUROC (needs only Node, no torch)
- `eval_fastdetect.py` — same eval for anyone with a local torch env

## Deploy (one time, free)
1. Create a Space: https://huggingface.co/new-space → SDK **Docker**, hardware
   **CPU basic (free)**. Needs a free HuggingFace account.
2. Push these `backend/` files to the Space repo:
   ```bash
   git clone https://huggingface.co/spaces/<user>/fast-detectgpt
   cp backend/{fast_detect.py,app.py,Dockerfile,requirements.txt,README.md} fast-detectgpt/
   cd fast-detectgpt && git add . && git commit -m "Fast-DetectGPT service" && git push
   ```
3. Wait for the build (first build downloads torch + warms gpt2, ~5 min).
4. Health check: open `https://<user>-fast-detectgpt.hf.space/` → `{"status":"ok"}`.

## Validate BEFORE wiring into index.html
```bash
node backend/eval_via_api.js https://<user>-fast-detectgpt.hf.space --limit 100
```
- If mean per-source AUROC clearly beats **0.524**, calibrate `CENTER`/`SCALE`
  in `app.py` (set CENTER to the discrepancy at the 5%-FPR threshold) and then
  add the opt-in "Enhanced Analysis" fetch + toggle to `index.html`.
- If it does **not** beat 0.524, do not deploy to the product — stylometry +
  a non-generalizing LM gains nothing.

## Notes
- Free Spaces sleep after ~48h idle; first request after sleep cold-starts (~30s).
- Endpoint is public and unauthenticated — fine, no secrets, public model on public text.
