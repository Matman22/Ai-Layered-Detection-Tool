# Research: Modern AI-Text Detection Techniques

*Compiled June 2026 — research branch. Maps recent (2025–2026) academic detection
methods against this project's current detection stack and specs the two highest-ROI
additions.*

---

## 0. Current Detection Approach (baseline for this analysis)

The tool runs four active layers, combined as a weighted evidence sum
(`index.html` for the live JS app; `raid_eval/detector.py` for the benchmarked port).

| Layer | What it does | How it works |
|-------|-------------|--------------|
| **L1 — Linguistic (18 signals)** | Statistical surface features | Burstiness, lexical diversity (TTR), AI-phrase lists, register/formality variance, paragraph & punctuation uniformity, vocab clustering, n-gram repetition, opener diversity, density-melody ensemble, Monte-Carlo windowing |
| **L2 — Forensic** | Character-stream artifacts | 23 invisible-Unicode types, homoglyphs, quote/dash/whitespace mixing |
| **L3 — Metadata** | File provenance | DOCX/PDF author, edit time, rsid sessions, timestamps, producer |
| **L5 — Authorial consistency** | Micro-habit drift | CV of contraction rate, Oxford comma, number formatting, etc. |

**Key facts that drive this research:**

- **The "perplexity" pillar is dead.** Every perplexity proxy (word-length variance,
  syllable CV, trigram entropy) is admittedly non-discriminating; its weight is cut to
  **0.00** in `detector.py` and ~0.01 in the JS app. The tool has *no real probabilistic
  language-model signal* despite "perplexity" being the headline concept in the field.
- **Almost every L1 signal measures the same thing:** low variance / high uniformity.
  Burstiness, paragraph uniformity, punctuation variance, vocab clustering, register
  stability, density melody — these are all CV-of-something. They are highly correlated,
  collapse together on short or single-paragraph text, and a single "vary your rhythm"
  evasion degrades all of them at once.
- **Recall is the weak point.** On the `andythetechnerd03/AI-human-text` (informal
  student-essay) set, the JS tool scores **48.9 F1 / 43.7% recall** — it confirms humans
  (79.8% specificity) far better than it catches AI. The Python port hits ~88% F1 on the
  easier RAID-style set, but that is a different, more formal distribution.
- **No learned model.** All 18 weights are hand-tuned. Roadmap Tier-3 Feature 10 already
  names this gap.
- **A backend is on the roadmap** (per project plan: 7-phase path to a FastAPI backend).
  Client-side GPT-2 was explicitly **deferred** (50–200 MB download + 10–40× slowdown),
  so any language-model-based signal belongs server-side, not in the browser bundle.

---

## 1. Five Recent Academic Approaches

For each: what it detects, how the algorithm works, reported accuracy, and client-side-JS
feasibility (the project's hard constraint).

---

### 1.1 Fast-DetectGPT — conditional probability curvature

- **Venue / date:** ICLR 2024 (arXiv Oct 2023). *The* canonical zero-shot baseline that
  every 2025–2026 paper benchmarks against — included as the reference probabilistic method.
- **What it detects:** Any machine-generated text, model-agnostic, zero-shot (no training).
- **How it works:** LLM-sampled text sits in a region where the model's log-probability is
  *locally maximal* relative to nearby token choices. Fast-DetectGPT scores a passage by its
  **conditional probability curvature**: for each position, sample alternative tokens from
  the scoring model's own conditional distribution (one forward pass, no perturb-and-rescore),
  then measure how much the passage's actual log-prob exceeds the expected log-prob of those
  alternatives, normalized by their standard deviation. AI text → high positive curvature;
  human text → curvature near zero. This replaces DetectGPT's ~100 perturbation passes with
  a single pass (≈340× faster).
- **Reported accuracy:** White-box AUROC ≈ **0.9887** across GPT-2/GPT-Neo/LLaMA sources
  (vs DetectGPT 0.9554). Degrades in the black-box setting and under heavy paraphrase.
- **Related:** **Binoculars** (ICML 2024) is the sibling method — it contrasts perplexity
  from an "observer" LM against cross-perplexity from a "performer" LM, hitting **>90%
  detection at a 0.01% false-positive rate**. Stronger FPR control but needs **two** LMs
  loaded simultaneously.
- **Client-side JS feasibility:** ⚠️ **Backend only.** Needs a real scoring LM
  (distilgpt2/GPT-2 ≈ 80–500 MB) and a per-token forward pass. Technically runnable via
  `transformers.js`/ONNX-WebGPU, but conflicts with the project's deferred-GPT-2 decision.
  Binoculars (two models) is doubly infeasible in-browser.

---

### 1.2 DivEye — rhythmic surprisal divergence

- **Venue / date:** 2025 (arXiv 2509.18880).
- **What it detects:** Machine text, with an explicit focus on **paraphrase- and
  domain-robustness** — the exact failure mode of curvature methods.
- **How it works:** Computes per-token **surprisal** (−log p) from a small LM, then throws
  away the mean and studies the *rhythm* of surprisal across the sequence — how
  unpredictability rises and falls. It extracts interpretable statistical features over the
  surprisal series (variance, autocorrelation / temporal structure, distributional shape at
  multiple scales). Core hypothesis: **human writing has richer variability in lexical and
  structural unpredictability**; LLM output is rhythmically flatter even after paraphrasing.
- **Reported accuracy:** Outperforms zero-shot detectors by **up to 33.2%**; improves
  existing detectors by **up to 18.7%** when added as an auxiliary signal; competitive with
  fine-tuned baselines; robust to paraphrase and adversarial edits.
- **Client-side JS feasibility:** ⚠️ **Hybrid.** The statistical layer is trivial JS, but it
  still needs an LM to produce per-token surprisal. Shares the *same forward pass* as
  Fast-DetectGPT — so if a backend already runs GPT-2 for curvature, DivEye is nearly free
  to add. Conceptually it is the rigorous, paraphrase-robust version of the project's
  hand-rolled "density melody" idea.

---

### 1.3 Intrinsic-Dimension / Persistent-Homology Dimension (PHD)

- **Venue / date:** NeurIPS 2023 origin, with multiple **2025 extensions** (e.g.
  cross-domain creative-story work, arXiv 2511.15210).
- **What it detects:** Machine text via the **geometry** of its embedding cloud — fully
  orthogonal to surface statistics and to perplexity.
- **How it works:** Embed every token/sentence with an encoder, then estimate the
  **intrinsic dimension** of the resulting point cloud using a persistent-homology dimension
  estimator. Empirically, **human text ≈ 9–10 intrinsic dimensions; AI text ≈ 8** — human
  writing explores a higher-dimensional manifold. Threshold on the estimate.
- **Reported accuracy:** Robust across generators and domains (the headline selling point);
  paper-dependent AUROC but consistently strong cross-model because it needs no knowledge of
  the generator. More stable under paraphrase than likelihood methods.
- **Client-side JS feasibility:** ⚠️ **Hard.** Needs a sentence/token **embedding model**
  (a small encoder, ~20–90 MB via `transformers.js` is plausible) *plus* a persistent-
  homology computation (no mature JS library — would need a hand-port of a TDA dimension
  estimator). Highest engineering cost of the five; best left as a research curiosity unless
  the backend exists.

---

### 1.4 NEULIF — lightweight stylometric classifier

- **Venue / date:** 2025 (arXiv 2511.21744).
- **What it detects:** AI vs human via **classic stylometric + readability features fed to a
  tiny trained model** — no language model at all.
- **How it works:** Extract ~**68 stylometric/readability features** (sentence-length stats,
  punctuation rates, function-word ratios, readability indices, lexical richness, etc.) and
  feed them to either a compact CNN or a Random Forest. <10⁵ parameters.
- **Reported accuracy:** On the Kaggle AI-vs-Human corpus — **CNN: 97% acc / ~0.95 F1 /
  0.995 ROC-AUC; Random Forest: 95% acc / ~0.94 F1 / 0.95 ROC-AUC.** (Caveat: the Kaggle set
  is known to be easy/synthetic; real-world numbers will be lower — but the *architecture* is
  the point.)
- **Client-side JS feasibility:** ✅ **Fully client-side.** RF ≈ 10.6 MB, CNN ≈ 25 MB, runs
  on CPU, no LLM. Critically, **this project already computes ~18 of these features.**
  Replacing hand-tuned weights with a model trained on those features is the single most
  natural upgrade — it is Roadmap Feature 10, made concrete.

---

### 1.5 Luminol-AIDetect — perplexity under text shuffling

- **Venue / date:** 2026 (arXiv 2604.25860). Newest of the set.
- **What it detects:** Machine text across domains, languages, and adversarial attacks;
  model-agnostic, zero-shot.
- **How it works:** Shuffle the text (sentence/segment order), measure perplexity of original
  vs shuffled, and use the **perplexity dispersion shift** as the discriminant. LLMs' strong
  local autoregressive consistency makes their perplexity respond to shuffling differently
  than human text. Scalar features from original+shuffled passes feed a density-estimation /
  ensemble classifier.
- **Reported accuracy:** Claims SOTA with **up to 17× lower false-positive rate** than prior
  methods, and cheaper to compute (no per-token perturbation loop). Abstract gives no single
  AUROC figure.
- **Client-side JS feasibility:** ⚠️ **Backend only**, same as 1.1 — still needs an LM for the
  perplexity passes, though shuffling itself is free. Cheaper than curvature at inference.

---

### Summary table

| # | Method | Year | Signal family | Reported acc. | Client-side JS |
|---|--------|------|---------------|---------------|----------------|
| 1 | Fast-DetectGPT (+Binoculars) | 2024 | LM probability curvature | AUROC ~0.99 / >90%@0.01% FPR | ❌ needs 1–2 LMs |
| 2 | DivEye | 2025 | LM surprisal rhythm | +33% vs zero-shot | ⚠️ hybrid (1 LM + light stats) |
| 3 | Intrinsic Dimension / PHD | 2023→2025 | Embedding geometry | robust cross-model | ❌ encoder + TDA, hard |
| 4 | NEULIF stylometric | 2025 | Trained feature classifier | 97% / 0.95 F1 | ✅ fully, no LLM |
| 5 | Luminol-AIDetect | 2026 | LM perplexity-under-shuffle | 17× lower FPR | ❌ needs 1 LM |

---

## 2. Blind-Spot Analysis — what the current layers miss

| Current blind spot | Why it exists | Which technique fills it |
|--------------------|---------------|--------------------------|
| **No real probabilistic signal** — the perplexity pillar is weighted to ~0 | Hand-rolled proxies don't approximate true token perplexity | **Fast-DetectGPT** / **Luminol** (true LM perplexity); **DivEye** (LM surprisal) |
| **Correlated uniformity signals** collapse together; one evasion beats all | ~10 of 18 L1 signals are CV-of-something | **NEULIF** (a trained model down-weights redundant features automatically); **Intrinsic Dimension** (orthogonal geometric signal) |
| **Low recall on informal / paraphrased AI** (43.7%) | Surface heuristics tuned for formal AI prose | **DivEye** (explicitly paraphrase-robust); **Fast-DetectGPT** (content-independent) |
| **Hand-tuned weights, no learning** | No training pipeline shipped to the app | **NEULIF** — directly converts existing features into learned weights |
| **Short / single-paragraph text** kills most signals | Paragraph/variance signals need 3+ paragraphs | **Fast-DetectGPT / DivEye** work per-token, no paragraph structure needed |
| **No robustness to humanizers/paraphrase attacks** | No signal targets the *process* of evasion | **DivEye**, **Intrinsic Dimension** (both reported paraphrase-robust) |

**Verdict.** Two gaps dominate: (a) there is *no learned model* even though the features
already exist, and (b) there is *no genuine language-model signal* even though "perplexity" is
the project's headline concept. The two implementation specs below target exactly these,
ordered by ROI × feasibility for a client-side-first tool with a planned backend.

---

## 3. Implementation Specs — Top 2 Techniques

### TOP 1 — Trained Stylometric Classifier (NEULIF-style) — *fully client-side, do first*

**Why first:** Highest ROI, zero new infrastructure, respects the no-server constraint. The
project already computes ~18 features in `detector.py`; this replaces the hand-tuned `WEIGHTS`
array with a model *trained on those same features*. It is Roadmap Feature 10 made concrete and
should measurably lift the 48.9 F1.

**What the code needs to do**

1. **Training (offline, Python — reuse `raid_eval/`):**
   - For every labeled sample, run the existing 18 `calc_*` functions to produce an
     18-dim feature vector. Optionally add a handful of cheap NEULIF features the tool lacks
     (Flesch-Kincaid, Gunning-Fog, function-word ratio, comma-rate) to widen the set.
   - Train **logistic regression** (interpretable, exports to ~20 floats) as the baseline; try
     a Random Forest as an upper bound.
   - Export the model as plain JSON (`weights[]`, `bias`, per-feature `mean`/`std` for
     standardization).

2. **Inference (JS, in `index.html`):**
   - Compute the same feature vector at runtime (functions already exist).
   - Standardize, dot-product with weights, apply sigmoid → AI probability.
   - Feed that probability into the existing combined-score model as a new high-weight L1
     input (or as a replacement aggregator for the L1 block).

**Inputs / outputs**

```
TRAIN:  texts[] + labels[]  ──►  model.json { feature_names[], mean[], std[], weights[], bias }
INFER:  rawText  ──►  featureVector[18+]  ──►  P(AI) ∈ [0,1]  ──►  score 0–100
```

**Pseudocode**

```python
# --- train.py (offline) ---
X = [[calc_burstiness(t), calc_lexical_diversity(t), ... calc_vocab_clustering(t),
      flesch_kincaid(t), function_word_ratio(t)]            # ~20 features
     for t in texts]
mean, std = column_mean(X), column_std(X)
Xz = (X - mean) / std
model = LogisticRegression().fit(Xz, labels)
dump_json({ "mean": mean, "std": std,
            "weights": model.coef_, "bias": model.intercept_ })
```

```javascript
// --- index.html (runtime) ---
const M = MODEL_JSON;                       // baked in, ~1 KB
function classifierScore(text) {
  const f = computeFeatureVector(text);     // existing calc_* functions
  let z = M.bias;
  for (let i = 0; i < f.length; i++)
    z += M.weights[i] * ((f[i] - M.mean[i]) / M.std[i]);
  const pAI = 1 / (1 + Math.exp(-z));        // sigmoid
  return pAI * 100;                          // 0–100, drop into combined score
}
```

**Success criterion:** re-run `evaluate.py` on the held-out informal set; F1 must beat the
current 48.9 (target ≥ 60) without specificity falling below the current 79.8%.

**Risks:** Kaggle-style training data overfits to easy synthetic text — train/validate on the
*informal* `andythetechnerd03` split, not the easy one, or numbers will look great and
generalize poorly.

---

### TOP 2 — Fast-DetectGPT Curvature + DivEye Surprisal (shared LM pass) — *backend / Phase 4*

**Why second & why paired:** This fills the dead perplexity pillar with a *real* signal,
orthogonal to every existing uniformity heuristic. Fast-DetectGPT and DivEye both consume the
**same single GPT-2 forward pass** (per-token log-probs), so shipping them together roughly
doubles the signal for ~no extra compute. Because client-side GPT-2 was deferred (size +
slowdown), this lives on the **planned FastAPI backend**; the browser calls it and folds the
result in as a new "Layer 4" when the network is available, degrading gracefully to L1/2/5
offline.

**What the code needs to do**

1. Load a small scoring LM once (e.g. `distilgpt2`, ~80 MB / `gpt2`, ~500 MB) on the backend.
2. One forward pass over the passage → per-token log-prob `lp[i]` and the full conditional
   distribution at each position.
3. **Fast-DetectGPT score** = curvature: `(logProb(text) − E[logProb(samples)]) / std`, where
   the expectation/std come from each position's own conditional distribution (closed-form, no
   actual resampling needed).
4. **DivEye score** from the *same* `surprisal[i] = −lp[i]` series: variance, lag-1
   autocorrelation, and distribution shape → small feature vector → calibrated to 0–100.
5. Return `{ curvatureScore, divEyeScore }`; the JS app blends them into the combined verdict.

**Inputs / outputs**

```
POST /analyze  { text }
  └─► one LM forward pass → lp[], condDist[]
        ├─ curvature  → fastDetectScore 0–100
        └─ surprisal[] = −lp[]  → {var, autocorr, shape} → divEyeScore 0–100
  ◄─ { fastDetectScore, divEyeScore }      # JS merges into Layer-4 input
```

**Pseudocode**

```python
# --- backend: layer4.py ---
def layer4(text):
    lp, dist = lm_forward(text)            # one pass: log-probs + conditional dists
    surprisal = [-x for x in lp]

    # Fast-DetectGPT: conditional probability curvature (closed form)
    mu  = [expected_logprob(d)  for d in dist]   # E over each position's own dist
    sig = [std_logprob(d)       for d in dist]
    curvature = mean((lp[i] - mu[i]) / sig[i] for i in range(len(lp)))
    fast_score = calibrate(curvature)            # AI → high

    # DivEye: rhythm of surprisal (reuses surprisal[], no extra LM cost)
    feats = [variance(surprisal),
             autocorr(surprisal, lag=1),
             kurtosis(surprisal)]
    div_score = calibrate_diveye(feats)          # flat rhythm → AI → high

    return { "fastDetectScore": fast_score, "divEyeScore": div_score }
```

```javascript
// --- index.html: optional Layer 4 ---
async function runLayer4(text) {
  try {
    const r = await fetch(BACKEND + "/analyze",
                          { method:"POST", body: JSON.stringify({ text }) });
    const { fastDetectScore, divEyeScore } = await r.json();
    return 0.6 * fastDetectScore + 0.4 * divEyeScore;   // L4 composite 0–100
  } catch {
    return null;   // offline → skip L4, reweight L1/L2/L5 to sum to 1
  }
}
```

**Success criterion:** on the informal split, adding L4 must lift recall by ≥ 10 points with
no more than a few points of specificity loss; ablation must show L4 is *not* redundant with
the L1 uniformity block (low correlation with existing signals).

**Risks:** (a) latency/cost of a backend LM call per analysis — cache by text hash; (b)
calibration of raw curvature → 0–100 must be fit on held-out data, not guessed; (c) black-box
gap — the scoring LM differs from the generator, which lowers real-world AUROC below the ~0.99
white-box figure. DivEye's paraphrase robustness partly offsets this.

---

## 4. Recommendation

1. **Ship TOP 1 (trained stylometric classifier) now.** Pure client-side, reuses existing
   features, directly attacks the "no learned model" and "correlated signals" blind spots, and
   is verifiable against the existing eval harness. This is the highest-confidence win.
2. **Build TOP 2 (Fast-DetectGPT + DivEye) as Layer 4 when the FastAPI backend lands.** It
   introduces the genuine probabilistic signal the tool has always lacked and is the only path
   to closing the recall gap on informal/paraphrased AI. Pairing the two methods on one LM pass
   is the efficient way to do it.
3. **Park Intrinsic Dimension and Luminol** as research extensions — strong ideas, but ID needs
   a TDA port with no JS ecosystem, and Luminol is strictly an LM-backend method that overlaps
   with Fast-DetectGPT's niche.

---

## Sources

- [Fast-DetectGPT (arXiv 2310.05130, ICLR 2024)](https://arxiv.org/abs/2310.05130)
- [Binoculars (arXiv 2401.12070, ICML 2024)](https://arxiv.org/abs/2401.12070)
- [DivEye: Diversity Boosts AI-Generated Text Detection (arXiv 2509.18880)](https://arxiv.org/pdf/2509.18880)
- [Intrinsic Dimension Estimation for Robust Detection (arXiv 2306.04723, NeurIPS 2023)](https://arxiv.org/pdf/2306.04723)
- [Unveiling Intrinsic Dimension of Texts, 2025 extension (arXiv 2511.15210)](https://arxiv.org/pdf/2511.15210)
- [NEULIF: Lightweight Stylometric Detection (arXiv 2511.21744)](https://arxiv.org/abs/2511.21744)
- [Luminol-AIDetect (arXiv 2604.25860)](https://arxiv.org/abs/2604.25860)
- [Adversarial Paraphrasing: A Universal Attack (arXiv 2506.07001)](https://arxiv.org/abs/2506.07001)
- [Why Perplexity and Burstiness Fail to Detect AI — Pangram Labs](https://www.pangram.com/blog/why-perplexity-and-burstiness-fail-to-detect-ai)
