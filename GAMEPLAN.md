# AI Detector — Build Gameplan

**Goal:** Evolve from a heuristic-only browser tool into a production-quality
Python detector backed by real language model signals, exposed via a FastAPI
endpoint, with the JS site calling it as an optional backend.

**Why Python:** Real perplexity (GPT-2), RAIDAR rewrite-analysis (Claude API),
and weight optimization via scikit-learn are not feasible in a browser.
Heuristics alone plateau at ~87–90% accuracy on RAID; LM-based signals push
past 95%.

---

## Benchmark Targets

| Phase | Recall | Precision | F1 |
|---|---|---|---|
| Current Python port | 87.6% | 92.7% | 0.843 |
| After Phase 1 (signal fixes) | ~90% | ~93% | ~0.915 |
| After Phase 2 (real perplexity) | ~92% | ~92% | ~0.92 |
| After Phase 3 (RAIDAR) | ~95% | ~93% | ~0.94 |
| After Phase 4 (weight optimization) | ~96% | ~94% | ~0.95 |

All targets measured on `liamdugan/raid` abstracts split, 500 samples.

---

## Architecture (End State)

```
User pastes text
        │
        ▼
┌──────────────────────────────────────────┐
│           JS Frontend (GitHub Pages)     │
│  • Local heuristics (instant, private)   │
│  • "Enhance with API" toggle             │
└──────────────┬───────────────────────────┘
               │ POST /analyze
               ▼
┌──────────────────────────────────────────┐
│         FastAPI Backend (Python)         │
│                                          │
│  Layer 1 — Heuristics (16 signals)       │
│  Layer 2 — Forensic character scan       │
│  Layer 3 — Real GPT-2 Perplexity         │
│  Layer 4 — RAIDAR (Claude API rewrite)   │
│  Layer 5 — Authorial consistency         │
│                                          │
│  Ensemble → weighted score → verdict     │
│  Returns: score, verdict, per-signal     │
└──────────────────────────────────────────┘
```

---

## Phase 1 — Fix Broken Signals
**Goal:** Fix the three signals identified as broken by RAID diagnostics before
adding anything new. These are bugs, not missing features.

**Files:** `raid_eval/detector.py`, `index.html`

### Checklist

- [ ] **Audit `rare_words`**
  - JS TP avg = 24.6, Python TP avg = 51.4 — 26.8 point gap means one is inverted
  - Compare JS `calcRareWords()` vs Python `calc_rare_words()` line by line
  - The correct behavior: AI text scores HIGHER (more rare/formal words)
  - Fix whichever implementation has the wrong polarity
  - Re-run RAID eval to confirm gap closes

- [ ] **Audit `sentence_opener_diversity`**
  - JS TP avg = 8.0, Python TP avg = 37.3 — 29.3 point gap
  - Compare JS `calcSentenceOpenerDiversity()` vs Python `calc_sentence_opener_diversity()`
  - The correct behavior: AI scores HIGHER (repetitive openers like "The", "This", "In")
  - Align formulas so both produce consistent values on same text

- [ ] **Audit `clause_depth` in JS**
  - JS FN avg = 74.7 vs JS TP avg = 41.7 — fires HARDER on missed AI than caught AI
  - Signal is inverting recall: high clause depth is being scored as "more AI" but
    the AI texts we miss (MPT/Llama-chat) use more complex clause structures
  - Options: remove from JS, invert the scoring direction, or reduce its weight to ~0.01
  - Re-run live_eval.py to confirm recall improvement

- [ ] **Re-run RAID eval on both versions after fixes**
  ```bash
  cd raid_eval
  python evaluate.py --n 500 --seed 42
  python live_eval.py --n 500
  ```
  Target: Python F1 > 0.90, JS false-positive rate below 50%

---

## Phase 2 — Real Perplexity (GPT-2)
**Goal:** Replace the 3-proxy perplexity approximation with a real language
model perplexity score. GPT-2 is free, runs locally, and is the standard
baseline for AI detection.

**New file:** `detector_v2/perplexity.py`

### How It Works
GPT-2 assigns a probability to every token. AI text is written *by* an LM so
it stays in high-probability regions — perplexity is low. Human text wanders
into low-probability regions — perplexity is high.

```
Human text → High perplexity (unpredictable word choices)
AI text    → Low perplexity  (predictable, stays on-manifold)
```

### Checklist

- [ ] **Install dependencies**
  ```bash
  pip install torch transformers accelerate
  ```

- [ ] **Implement `perplexity.py`**
  ```python
  from transformers import GPT2LMHeadModel, GPT2TokenizerFast
  import torch

  def gpt2_perplexity(text: str, stride: int = 512) -> float:
      """Sliding-window perplexity using GPT-2. Lower = more AI-like."""
      # Load model once and cache
      # Tokenize, slide window, average NLL
      # Return normalized 0-100 score (invert so higher = more AI)
  ```

- [ ] **Sliding window approach** (handles texts longer than GPT-2's 1024 token limit)
  - Window size: 512 tokens, stride: 256
  - Average negative log-likelihood across windows
  - Normalize against calibration corpus (RAID human samples)

- [ ] **Score normalization**
  - Run on 200 RAID human samples → establish "human baseline" perplexity
  - Run on 200 RAID AI samples → establish "AI baseline"
  - Map onto 0–100 scale: 100 = definitely AI, 0 = definitely human

- [ ] **Add to detector ensemble**
  - Add as `layer3_perplexity` signal
  - Initial weight: 0.20 (high — this is a real signal, not a proxy)
  - Reduce heuristic weights proportionally to keep sum = 1.0

- [ ] **Evaluate improvement**
  ```bash
  python evaluate.py --n 500 --seed 42
  ```
  Target: recall > 92%, F1 > 0.92

- [ ] **Add `--fast` flag** to skip perplexity for quick runs (heuristics only)

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
  - `full` — heuristics + GPT-2 perplexity, ~2s, no API cost
  - `raidar` — all signals including Claude rewrite, ~5s, ~$0.001/call

- [ ] **Response schema**
  ```json
  {
    "score": 78,
    "verdict": "AI",
    "confidence": "high",
    "signals": {
      "burstiness": 74.2,
      "formality_shift": 91.5,
      "gpt2_perplexity": 82.1,
      "raidar": 88.0,
      ...
    },
    "layers": {
      "l1_heuristics": 71,
      "l2_forensic": 0,
      "l3_perplexity": 82,
      "l4_raidar": 88,
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
├── index.html                  ← JS frontend (live site)
├── GAMEPLAN.md                 ← this file
├── README.md                   ← updated with benchmarks
├── CLAUDE.md                   ← codebase context
│
├── detector_v2/                ← Python detector (production)
│   ├── __init__.py
│   ├── detector.py             ← main scorer (all signals)
│   ├── heuristics.py           ← ported + fixed JS signals
│   ├── perplexity.py           ← GPT-2 perplexity (Phase 2)
│   ├── raidar.py               ← Claude rewrite signal (Phase 3)
│   ├── forensic.py             ← character-level forensics
│   ├── authorial.py            ← Layer 5 micro-habits
│   └── optimize_weights.py     ← sklearn weight tuning (Phase 4)
│
├── api/                        ← FastAPI backend (Phase 5)
│   ├── main.py
│   ├── models.py
│   ├── requirements.txt
│   └── Dockerfile
│
└── raid_eval/                  ← benchmarking tools (existing)
    ├── detector.py             ← original Python port
    ├── evaluate.py             ← RAID benchmark runner
    ├── diagnose.py             ← signal diagnostics
    ├── live_eval.py            ← JS vs RAID
    ├── compare_versions.py     ← JS vs Python comparison
    └── requirements.txt
```

---

## Quick-Start Checklist (Do These First)

Before writing any new code, fix the existing bugs:

```bash
# 1. Audit and fix the three broken signals
#    rare_words / sentence_opener_diversity / clause_depth

# 2. Re-run to confirm fixes
cd raid_eval
python evaluate.py --n 500

# 3. Then start Phase 2 (perplexity)
pip install torch transformers
```

---

## Effort Estimates

| Phase | Estimated Time | Complexity |
|---|---|---|
| 1 — Fix signals | 2–3 hours | Low |
| 2 — GPT-2 perplexity | 4–6 hours | Medium |
| 3 — RAIDAR | 3–4 hours | Medium |
| 4 — Weight optimization | 2–3 hours | Low |
| 5 — FastAPI | 4–6 hours | Medium |
| 6 — Deploy | 1–2 hours | Low |
| 7 — JS frontend update | 3–4 hours | Medium |
| 8 — Documentation | 2–3 hours | Low |
| **Total** | **~25–30 hours** | |

---

*Last updated: 2026-05-14*
*Benchmarks measured on liamdugan/raid, abstracts split, 500 samples, seed=42*
