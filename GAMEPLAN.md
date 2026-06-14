# AI Detector — Build Gameplan

**Goal:** Evolve from a heuristic-only browser tool into a multi-model AI
detector that blends stylometric analysis, neural classifiers, and real
language model signals — all at zero cost to the developer and end user.

**Strategy:** Add neural detection in two tiers: (1) RoBERTa AI detector
running client-side via Transformers.js/ONNX (no server needed), and
(2) GPT-2 perplexity on a free HuggingFace Space (opt-in, server-side).
Later phases add RAIDAR rewrite-analysis and weight optimization via a
FastAPI backend.

**Why this order:** The RoBERTa detector and GPT-2 perplexity are the two
highest-ROI additions available at $0. They add real neural/LM signals that
stylometric features physically cannot capture. RAIDAR requires an API key
with per-call cost, so it comes after the free options are exhausted.

---

## Benchmark Targets

| Phase | F1 | Notes |
|---|---|---|
| Phase 1 — Signal parity | 0.905 | ✅ Complete |
| Phase 1b — Top-1 classifier | 0.916 | ✅ Complete (ML_WEIGHTS, 20-signal LR) |
| Phase 2a — RoBERTa (client-side) | ~0.93 | Next: Transformers.js, $0 |
| Phase 2b — GPT-2 perplexity (server) | ~0.94 | Next: HF Space, $0 |
| Phase 3 — RAIDAR | ~0.95 | Claude API, ~$0.001/call |
| Phase 4 — Weight optimization | ~0.95 | sklearn retrain on all signals |

All targets measured on `liamdugan/raid` abstracts split, 500 samples, seed=42.

---

## Architecture (End State)

```
User pastes text
        │
        ▼
┌──────────────────────────────────────────────────┐
│       Tier 1 — Instant, private (client-side)    │
│                                                  │
│  L1: Stylometric classifier (17 signals, LR)     │
│  L2: Forensic Unicode scan                       │
│  L3: File metadata (DOCX/PDF)                    │
│  L5: Authorial consistency                       │
│  L6: RoBERTa AI detector (ONNX, Transformers.js) │ ← Phase 2a
│                                                  │
│  Verdict from ML_WEIGHTS (20-signal LR)          │
│  + RoBERTa probability blended in                │
└──────────────┬───────────────────────────────────┘
               │ opt-in "Enhanced Analysis" toggle
               ▼
┌──────────────────────────────────────────────────┐
│   Tier 2 — Enhanced, free (HuggingFace Space)    │
│                                                  │
│  L7: GPT-2 token perplexity (real, not proxy)    │ ← Phase 2b
│  L7b: Fast-DetectGPT curvature (same GPT-2 pass) │
│                                                  │
│  ~2-5s latency, 30s cold start after idle        │
│  Free forever (HF Spaces CPU tier)               │
└──────────────┬───────────────────────────────────┘
               │ future (requires API key)
               ▼
┌──────────────────────────────────────────────────┐
│      Tier 3 — Premium (FastAPI backend)          │
│                                                  │
│  L8: RAIDAR rewrite analysis (Claude API)        │ ← Phase 3
│  Weight-optimized ensemble over all layers       │ ← Phase 4
│                                                  │
│  Deployed on Render / HF Spaces                  │
└──────────────────────────────────────────────────┘
```

---

## Phase 1 — Fix Broken Signals ✅ COMPLETE
**Status:** Signal parity aligned. Top-1 classifier shipped (logistic regression
over 17 signals → `classifierL1()` in `index.html`). Phase 3 ML_WEIGHTS model
(20-signal LR, F1 0.916 on RAID) also shipped and drives the final combined score.

See `RESEARCH.md` for the 5-technique survey and implementation specs.
See `eval/train_classifier.js` and `eval/model.json` for the training pipeline.

---

## Phase 2a — RoBERTa AI Detector (Client-Side, $0) ← NEXT
**Goal:** Add a real neural classifier that runs in the browser via
Transformers.js + ONNX. This catches token-level patterns that stylometric
signals cannot detect.

**Model:** `onnx-community/roberta-base-openai-detector-ONNX` (int8 quantized,
~126 MB, downloaded once and cached by the browser).

**Cost:** $0 — runs entirely client-side. No API key, no server, no rate limits.

**File:** `index.html`

### How It Works
```
User clicks "Analyze"
    │
    ├── L1-L5 run instantly (existing stylometric signals)
    │
    ├── RoBERTa loads (first time: ~5-10s download, then cached)
    │   └── Transformers.js pipeline("text-classification", model)
    │       └── Returns: { label: "LABEL_0/LABEL_1", score: 0.0-1.0 }
    │           LABEL_0 = human, LABEL_1 = AI (GPT-2 trained)
    │
    └── Blend RoBERTa probability into the ML_WEIGHTS ensemble
```

### Checklist

- [ ] **Add Transformers.js to index.html**
  - Load from CDN: `https://cdn.jsdelivr.net/npm/@huggingface/transformers`
  - Initialize pipeline lazily (only on first analysis, not on page load)
  - Show loading indicator: "Loading neural model (one-time ~126 MB download)..."

- [ ] **Run RoBERTa inference**
  ```javascript
  import { pipeline } from '@huggingface/transformers';
  const detector = await pipeline('text-classification',
    'onnx-community/roberta-base-openai-detector-ONNX',
    { dtype: 'q8' }  // int8 quantized
  );
  const result = await detector(text);
  // result: [{ label: 'LABEL_1', score: 0.92 }]
  const robertaScore = result[0].label === 'LABEL_1'
    ? result[0].score * 100
    : (1 - result[0].score) * 100;
  ```

- [ ] **Add as Layer 6 in the UI**
  - New panel: "Neural Analysis (RoBERTa)"
  - Show confidence: "92% probability of AI generation"
  - Toggle: "Enable neural detection (requires ~126 MB download)"
  - Graceful degradation: if user declines download, L1-L5 still work

- [ ] **Blend into ensemble**
  - Add `robertaScore` to the ML_WEIGHTS feature vector
  - Retrain logistic regression with RoBERTa as signal #21
  - Or: simple weighted blend `combinedScore * 0.6 + robertaScore * 0.4`

- [ ] **Evaluate improvement**
  - Score RAID test set with and without RoBERTa
  - Measure F1 lift, especially on GPT-family generators
  - Document: RoBERTa was trained on GPT-2 — expect lower lift on Claude/Llama

- [ ] **Performance budget**
  - First load: ~5-10s (126 MB download, WASM compilation)
  - Subsequent loads: ~1-2s (model cached, WASM cached)
  - Inference: ~500ms-2s per text (CPU, depends on length)
  - Target: total analysis under 3s after first load

### Caveat
RoBERTa-base-openai-detector was fine-tuned on GPT-2 output (2019). It
generalizes somewhat to GPT-3.5/4 but less reliably to Claude, Llama, and
Mistral. Still, it provides a fundamentally different signal class (token
distributions vs. surface statistics) that complements the existing ensemble.

---

## Phase 2b — GPT-2 Perplexity Backend (HuggingFace Space, $0)
**Goal:** Deploy GPT-2 (124M) on a free HuggingFace Space to compute real
token-level perplexity. This revives the "dead" perplexity signal (currently
weight=0.00 because it's a proxy, not real perplexity).

**Cost:** $0 forever. HF Spaces free CPU tier: 2 vCPUs, 16 GB RAM. Sleeps
after 48h idle, wakes on request (~30s cold start).

**New files:** `hf-space/app.py`, `hf-space/requirements.txt`

### How It Works
```
User enables "Enhanced Analysis" toggle
    │
    ▼
POST https://your-space.hf.space/api/perplexity
  body: { "text": "..." }
    │
    ▼
GPT-2 forward pass on HF Space (CPU, ~2-5s)
    │
    ├── Token-level log-probabilities
    ├── Perplexity = exp(mean negative log-prob)
    ├── Burstiness = std of per-token surprisal
    └── (Optional) Fast-DetectGPT curvature score
    │
    ▼
Returns: { perplexity, burstiness, curvature }
    │
    ▼
JS frontend blends into verdict (L7 signal)
```

### Checklist

- [ ] **Build Gradio app (`hf-space/app.py`)**
  ```python
  import gradio as gr
  import torch
  from transformers import GPT2LMHeadModel, GPT2Tokenizer

  model = GPT2LMHeadModel.from_pretrained("gpt2")
  tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

  def compute_perplexity(text: str):
      tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
      with torch.no_grad():
          outputs = model(**tokens, labels=tokens["input_ids"])
      perplexity = torch.exp(outputs.loss).item()
      # Per-token surprisal for burstiness
      logits = outputs.logits[:, :-1, :]
      targets = tokens["input_ids"][:, 1:]
      log_probs = -torch.nn.functional.cross_entropy(
          logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction='none'
      )
      burstiness = log_probs.std().item()
      return {"perplexity": round(perplexity, 2), "burstiness": round(burstiness, 4)}

  demo = gr.Interface(fn=compute_perplexity, inputs="text", outputs="json")
  demo.launch()
  ```

- [ ] **Deploy to HuggingFace Spaces**
  - Create Space: `huggingface.co/new-space` → Gradio, CPU, free
  - Push `app.py` + `requirements.txt` (torch, transformers, gradio)
  - Test the `/api/predict` endpoint

- [ ] **Add opt-in toggle in index.html**
  ```html
  <label class="api-toggle">
    <input type="checkbox" id="useEnhanced">
    Enhanced analysis (sends text to HuggingFace — more accurate, ~5s)
  </label>
  ```

- [ ] **Wire into scoring**
  - Call HF Space API from `runAnalysis()` when toggle is checked
  - Map perplexity to 0-100 score (low perplexity → high AI probability)
  - Add as L7 signal in the ensemble
  - Graceful fallback: if Space is sleeping/unreachable, skip and show
    "Enhanced analysis unavailable" notice

- [ ] **Implement Fast-DetectGPT curvature** (bonus, same GPT-2 pass)
  - Conditional probability curvature from the RESEARCH.md Top-2 spec
  - Adds a second signal from the same forward pass — free marginal cost

- [ ] **Evaluate improvement**
  - Score RAID test set with perplexity signal active vs. baseline
  - Target: F1 improvement on texts where stylometric signals are weak
    (paraphrased, short, or style-mimicked text)

### Privacy notice
When Enhanced Analysis is enabled, the user's text is sent to a HuggingFace
Space (your own, not a third party). The toggle must be off by default with
a clear label. No text is stored server-side.

---

## Phase 3 — RAIDAR (Rewrite-Based Detection)
**Goal:** Use the Claude API to rewrite the input text and measure how much it
changed. AI text gets barely rewritten (already fluent); human text gets
substantially rewritten.

**Reference:** "RAIDAR: gpt-4 is a zero-shot detector" (2024).
Your version uses Claude Haiku (cheap, fast) instead of GPT-4.

**New file:** `detector_v2/raidar.py`

### How It Works
```
Input text
    │
    ▼
Claude Haiku: "Rewrite this text to improve clarity and flow.
               Keep the same meaning and length."
    │
    ▼
Measure edit distance (character-level Levenshtein, normalized)
    │
AI text:    edit_distance ~0.05–0.15  (Claude barely changes it)
Human text: edit_distance ~0.30–0.60  (Claude substantially rewrites)
    │
    ▼
Score: 100 − (edit_distance × 200)  → higher = more AI
```

### Checklist

- [ ] **Install dependencies**
  ```bash
  pip install anthropic python-levenshtein
  ```

- [ ] **Implement `raidar.py`**
  ```python
  import anthropic
  from Levenshtein import ratio

  def raidar_score(text: str, client: anthropic.Anthropic) -> float:
      """
      Rewrites text with Claude Haiku and measures edit distance.
      Returns 0-100 where 100 = very AI-like (barely changed).
      """
      rewritten = client.messages.create(
          model="claude-haiku-4-5-20251001",
          max_tokens=len(text.split()) * 2,
          messages=[{
              "role": "user",
              "content": f"Rewrite the following text to improve clarity and natural flow. "
                         f"Keep the same meaning, length, and topic. Output only the rewritten text:\n\n{text}"
          }]
      ).content[0].text

      similarity = ratio(text, rewritten)        # 0.0 = totally different, 1.0 = identical
      ai_score = round(similarity * 100)         # high similarity = AI wrote it
      return min(100, max(0, ai_score))
  ```

- [ ] **Add caching** (avoid re-calling API on same text)
  - SHA256 hash of text as cache key
  - Store in `~/.cache/raidar_cache.json`

- [ ] **Add to detector ensemble**
  - Add as `layer4_raidar` signal
  - Initial weight: 0.25 (highest single signal weight)
  - Reduce other weights proportionally

- [ ] **Evaluate improvement**
  ```bash
  python evaluate.py --n 300 --seed 42    # smaller n due to API cost
  ```
  Target: recall > 95%, F1 > 0.94

- [ ] **Add `--no-raidar` flag** to skip for cost-free runs

- [ ] **Estimate API cost**
  - Claude Haiku input: $0.80/M tokens, output: $4.00/M tokens
  - ~500 token text = ~$0.001 per sample
  - 500 samples ≈ $0.50 total for a full RAID eval run

---

## Phase 4 — Weight Optimization
**Goal:** Use scikit-learn to find optimal signal weights from RAID labeled data
instead of hand-tuning.

**New file:** `detector_v2/optimize_weights.py`

### Checklist

- [ ] **Generate feature matrix from RAID**
  - Score 2000 RAID samples through all signals
  - Save as `features.csv` with columns: [signal_1 ... signal_N, label]

- [ ] **Fit logistic regression**
  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.model_selection import cross_val_score

  clf = LogisticRegression(C=1.0, max_iter=1000)
  scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
  ```

- [ ] **Extract learned weights**
  - Use `clf.coef_` as the new signal weights
  - Normalize to sum to 1.0
  - Compare learned weights vs current hand-tuned weights

- [ ] **Evaluate optimized weights**
  - Hold out 500 samples as test set (not used in training)
  - Report F1 improvement vs hand-tuned

- [ ] **Update `detector_v2/detector.py` with optimized weights**

---

## Phase 5 — FastAPI Backend
**Goal:** Expose the Python detector as a REST API so the JS frontend can
call it.

**New files:** `api/main.py`, `api/models.py`, `api/requirements.txt`

### Checklist

- [ ] **Install dependencies**
  ```bash
  pip install fastapi uvicorn pydantic python-dotenv
  ```

- [ ] **Implement `api/main.py`**
  ```python
  from fastapi import FastAPI
  from fastapi.middleware.cors import CORSMiddleware
  from pydantic import BaseModel

  app = FastAPI(title="AI Detector API")
  app.add_middleware(CORSMiddleware, allow_origins=["*"])

  class AnalyzeRequest(BaseModel):
      text: str
      mode: str = "full"   # "fast" | "full" | "raidar"

  @app.post("/analyze")
  async def analyze(req: AnalyzeRequest):
      # Route to appropriate detector based on mode
      ...

  @app.get("/health")
  def health(): return {"status": "ok"}
  ```

- [ ] **Three modes**
  - `fast` — heuristics only, <100ms, no API cost
  - `full` — heuristics + RAIDAR, ~5s, ~$0.001/call
  - `raidar` — alias for full mode (kept for backwards-compat)

- [ ] **Response schema**
  ```json
  {
    "score": 78,
    "verdict": "AI",
    "confidence": "high",
    "signals": {
      "burstiness": 74.2,
      "formality_shift": 91.5,
      "lexical_diversity": 85.0,
      "raidar": 88.0,
      ...
    },
    "layers": {
      "l1_heuristics": 71,
      "l2_forensic": 0,
      "l3_raidar": 88,
      "l5_authorial": 45
    },
    "mode": "full"
  }
  ```

- [ ] **Input validation**
  - Minimum 50 words, maximum 10,000 words
  - Strip HTML tags
  - Return 400 with helpful message if too short

- [ ] **Run locally and test**
  ```bash
  uvicorn api.main:app --reload
  curl -X POST http://localhost:8000/analyze \
       -H "Content-Type: application/json" \
       -d '{"text": "...", "mode": "fast"}'
  ```

- [ ] **Run RAID eval through the API** (end-to-end integration test)
  ```bash
  python raid_eval/evaluate.py --api http://localhost:8000/analyze
  ```

---

## Phase 6 — Deploy Backend
**Goal:** Get the API running at a public URL so the JS frontend can call it.

### Options (pick one)

| Platform | Free tier | Cold start | Best for |
|---|---|---|---|
| **Render** | 750 hrs/mo | ~30s | Easiest deploy |
| **Railway** | $5 credit | ~5s | Best DX |
| **HuggingFace Spaces** | Free (CPU) | ~10s | ML-friendly |
| **Fly.io** | 3 free VMs | ~2s | Best performance |

**Recommended: Render** (free, simple GitHub auto-deploy)

### Checklist

- [ ] **Add `Dockerfile`** (or `render.yaml`)
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY api/requirements.txt .
  RUN pip install -r requirements.txt
  COPY . .
  CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- [ ] **Add `ANTHROPIC_API_KEY` as environment secret** in hosting dashboard

- [ ] **Connect GitHub repo** → auto-deploys on push to `main`

- [ ] **Test live API URL**
  ```bash
  curl https://your-app.onrender.com/health
  ```

- [ ] **Add API URL to `GAMEPLAN.md` once deployed**

---

## Phase 7 — Update JS Frontend
**Goal:** Add an "Enhance with API" toggle to the live site that calls the
Python backend when enabled.

**File:** `index.html`

### Checklist

- [ ] **Add API toggle button** to the UI (below the analyze button)
  ```html
  <label class="api-toggle">
    <input type="checkbox" id="useApi">
    Use enhanced API (GPT-2 + RAIDAR) — more accurate, ~5s
  </label>
  ```

- [ ] **Add API call in `runAnalysis()`**
  ```javascript
  const API_URL = "https://your-app.onrender.com/analyze";

  if (document.getElementById('useApi').checked) {
    const res = await fetch(API_URL, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ text, mode: 'full' })
    });
    const apiResult = await res.json();
    // Merge API result into display
  }
  ```

- [ ] **Show both scores** side-by-side in the results UI
  - "Local (instant)": JS heuristic score
  - "Enhanced (API)": Python backend score

- [ ] **Graceful fallback** — if API is unreachable, show local score only
  with a "API unavailable — showing local result" notice

- [ ] **Update CLAUDE.md** with new architecture notes

- [ ] **Commit and push** → GitHub Pages auto-deploys

---

## Phase 8 — Documentation & Portfolio
**Goal:** Make this project presentable for co-op applications.

### Checklist

- [ ] **Update `README.md`**
  - Architecture diagram (ASCII or image)
  - Benchmark table vs RAID baseline
  - How to run locally (API + frontend)
  - API endpoint documentation

- [ ] **Write methodology section**
  - Explain each detection layer
  - Cite RAID paper, RAIDAR paper
  - Show before/after benchmark numbers

- [ ] **Compare to published baselines**
  - GLTR, Binoculars, DetectGPT accuracy on RAID
  - Your detector vs those numbers
  - Use `raid_eval/evaluate.py` results

- [ ] **Add architecture diagram** to README
  (ASCII diagram from this document, or generate a proper image)

- [ ] **Record a demo video** (optional but impactful)
  - 2-minute screen recording
  - Show AI text → high score, human text → low score
  - Show the API toggle in action

---

## File Structure (End State)

```
Ai-Layered-Detection-Tool/
│
├── index.html                  ← JS frontend (Tiers 1+2 toggle)
├── GAMEPLAN.md                 ← this file
├── RESEARCH.md                 ← 5-technique survey + specs
├── README.md                   ← updated with benchmarks
├── CLAUDE.md                   ← codebase context
│
├── eval/                       ← training pipeline (Node.js)
│   ├── extract_features.js     ← feature extraction via index.html fns
│   ├── train_classifier.js     ← pure-Node LR trainer
│   ├── model.json              ← exported Top-1 classifier weights
│   ├── features.csv            ← 500-row feature matrix
│   ├── fetch_dataset.js        ← HF REST dataset fetcher
│   └── *.csv                   ← multi-dataset CSVs
│
├── hf-space/                   ← GPT-2 perplexity backend (Phase 2b)
│   ├── app.py                  ← Gradio app for HuggingFace Spaces
│   └── requirements.txt        ← torch, transformers, gradio
│
├── detector_v2/                ← Python detector (Phase 3+)
│   ├── raidar.py               ← Claude rewrite signal
│   ├── heuristics.py           ← ported JS signals
│   └── optimize_weights.py     ← sklearn weight tuning (Phase 4)
│
├── api/                        ← FastAPI backend (Phase 5)
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── raid_eval/                  ← benchmarking tools (existing)
    ├── detector.py             ← reference Python implementation
    ├── evaluate.py             ← RAID benchmark runner
    ├── node_runner.js          ← Node.js scorer (evals index.html)
    └── requirements.txt
```

---

## Quick-Start Checklist (Do These First)

Phase 1 complete (signal parity + Top-1 classifier + ML_WEIGHTS).
Next step: Phase 2a (RoBERTa in-browser) and Phase 2b (GPT-2 HF Space).

```bash
# Phase 2a — RoBERTa (client-side, just edit index.html)
# No install needed — Transformers.js loads from CDN

# Phase 2b — GPT-2 perplexity (HuggingFace Space)
cd hf-space
pip install torch transformers gradio
python app.py  # local test, then push to HF Spaces
```

---

## Effort Estimates

| Phase | Estimated Time | Complexity | Cost |
|---|---|---|---|
| 1 — Signal parity | ~~4–6 hours~~ | ~~Medium~~ | ✅ Done |
| 2a — RoBERTa (client-side) | 4–6 hours | Medium | $0 |
| 2b — GPT-2 perplexity (HF Space) | 3–4 hours | Medium | $0 |
| 3 — RAIDAR | 3–4 hours | Medium | ~$0.001/call |
| 4 — Weight optimization | 2–3 hours | Low | $0 |
| 5 — FastAPI backend | 4–6 hours | Medium | $0 (free hosting) |
| 6 — Deploy | 1–2 hours | Low | $0 |
| 7 — JS frontend update | 3–4 hours | Medium | $0 |
| 8 — Documentation | 2–3 hours | Low | $0 |
| **Remaining** | **~23–32 hours** | | |

---

*Last updated: 2026-06-13*
*Benchmarks measured on liamdugan/raid, abstracts split, 500 samples, seed=42*
