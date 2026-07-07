# AI Origin Detector

A multi-layer forensic tool for detecting AI-generated text,
built entirely in vanilla JavaScript — no server, no API,
nothing sent anywhere.

**Live:** https://matman22.github.io/Ai-Layered-Detection-Tool/

---

## How it works

### Layer 1 — Linguistic Analysis (17 signals)
Perplexity modeling (3-proxy ensemble), sentence burstiness,
lexical diversity, AI phrase fingerprinting (T1/T2 tiered),
hedging density, passive voice, n-gram repetition, sentence
opener diversity, punctuation fingerprinting, vocabulary
clustering, density melody ensemble, Monte Carlo window
sampling, and more. Scored by a logistic-regression classifier
trained on labeled data (replacing the original hand-tuned
weights), rendered as an animated gauge with ranked
per-signal contribution bars.

### Layer 2 — Forensic Character Scan
Scans raw Unicode codepoints for invisible characters
(zero-width spaces, word joiners, soft hyphens), homoglyph
substitutions (Cyrillic/Greek chars disguised as Latin),
non-standard whitespace, and mixed quote encoding — artifacts
that survive editing and are invisible to the naked eye.

### Layer 3 — File Metadata Analysis
Reads DOCX and PDF metadata directly in the browser — author
fields, total editing time, revision session markers,
created/modified timestamps, and application signatures.
A document with 0 minutes of editing time tells a different
story than one with 187.

### Layer 5 — Authorial Consistency Fingerprinting
Measures micro-habit consistency across five signals: contraction
rate, Oxford comma usage, number formatting, sentence-final
prepositions, and paragraph opener word class. Scores the
*consistency* of each habit, not the habit itself — so formal
and casual writers both score human for different reasons.

---

## Evaluation: what stylometry can and can't do

The most important result of this project is a negative one,
found by measuring instead of assuming.

The classifier was evaluated on **five public datasets**
(RAID, MAGE, HC3, and two Kaggle/HF corpora, ~1,900 balanced
samples) fetched free through the HuggingFace datasets-server
API. In-distribution it looks strong (5-fold F1 ≈ 0.76). But
under **leave-one-source-out cross-validation** — train on four
datasets, test on the fifth — mean AUROC collapses to **0.52,
statistically indistinguishable from chance** (one dataset even
scores *inverted* at 0.26).

An automated optimization loop then ran 7 iterations of feature
engineering and selection-rule search. In-loop LOSO climbed to
0.67 — but a pre-registered validation checkpoint on a **sixth,
never-touched dataset** exposed the gain as meta-overfitting:
the "best" selection rule scored **0.45 on truly fresh data**,
and no configuration cleared 0.55. The external ranking of
configs nearly reversed the in-loop ranking.

**Conclusion:** surface stylometry learns domain quirks, not
authorship. Cross-domain detection needs a language-model
signal. The full experiment log (hypotheses, falsifications,
exact numbers) is in `eval/experiments/LOG.md`.

### Next: Fast-DetectGPT backend
`backend/` contains a ready-to-deploy FastAPI service
implementing **Fast-DetectGPT** (Bao et al., ICLR 2024) —
zero-shot conditional probability curvature on GPT-2. Because
it has no learned decision boundary, it should transfer across
domains where the trained classifier could not. Designed to run
on a free HuggingFace Space (Docker, CPU); it is gated on
beating the 0.52 stylometric ceiling on the same eval data
before being wired into the product.

---

## Methodology

This tool was built iteratively using an adversarial,
measurement-first approach:

1. Built initial 12-vector linguistic detector
2. Generated evasion text specifically crafted to fool it
3. Analyzed exactly which signals were bypassed and why
4. Patched weaknesses; replaced hand-tuned weights with a
   trained logistic-regression classifier
5. Added forensic and metadata layers targeting signals
   that survive stylistic evasion
6. Stress-tested generalization with multi-dataset LOSO
   evaluation and a pre-registered external validation set —
   and reported the honest result

---

## Repo structure

```
index.html          the entire app — runs client-side, no build step
eval/               dataset fetching, feature extraction, training,
                    LOSO AUROC evaluation (Node, zero dependencies)
eval/experiments/   the 7-iteration optimization loop: scripts,
                    datasets, and LOG.md with every result
backend/            Fast-DetectGPT FastAPI service (deployable to a
                    free HF Space) + API-based evaluation harness
raid_eval/          RAID benchmark scorer + Python training pipeline
```

## Tech

Pure HTML/CSS/JavaScript front end. No frameworks, no dependencies
(PDF.js loaded from CDN for PDF parsing). Runs entirely client-side.
Evaluation pipeline in plain Node; optional LM backend in
Python/FastAPI/PyTorch.

## Skills demonstrated

- NLP & computational linguistics
- Empirical ML evaluation: cross-dataset LOSO validation, AUROC,
  pre-registered analysis, catching meta-overfitting
- Adversarial ML / red-teaming methodology
- Digital forensics (Unicode, file metadata, binary parsing)
- Iterative system design from first principles

---

## Future work — v2: engineering for the RAID leaderboard

The current detector is optimized for **honest generalization** — working on
text it has never seen. A planned v2 would optimize for a different objective:
the **highest possible score on the [RAID benchmark](https://raid-bench.xyz)**,
which grades accuracy at a strict 5% false-positive rate across 8 domains,
11 generators, and ~11 adversarial attacks. These are different games — RAID
provides a labeled *training* split covering the same distribution as its test
set, so the winning strategy is on-distribution specialization, not generality.

Planned approach, in order of expected impact:

1. **Supervised transformer fine-tuned on RAID-train.** Fine-tune DeBERTa-v3 /
   RoBERTa on RAID's own training data so the model learns the exact generator
   and attack distribution it's graded on. This is the single biggest lever and
   fits a free Colab T4 GPU. (The current zero-shot approach caps far lower.)
2. **Input-normalization front-end.** Strip zero-width characters, map homoglyphs
   back to Latin, and collapse trick whitespace *before* the model sees the text —
   defeating several adversarial attack types cheaply. This repurposes the
   existing Layer 2 forensic code from a weak standalone detector into an
   adversarial defense, where it is genuinely valuable.
3. **Adversarial data augmentation.** Apply the same attacks (paraphrase, synonym
   swap, homoglyph) to training data to harden the model on those slices.
4. **Ensemble with a strong zero-shot signal** (Binoculars, or Fast-DetectGPT
   with larger open base models) to hedge against under-represented generators.
5. **Per-slice threshold calibration** to hit the 5%-FPR operating point exactly.

**Honest tradeoff:** a RAID-tuned model would score far higher on the leaderboard
but generalize *worse* to non-RAID text — a better number, a more specialized
detector. Carried in the repo (rather than done immediately) precisely because
the current generalist result, and the discipline of not overfitting to one
benchmark, is the more transferable engineering story.

Carried over from v1: the LOSO evaluation harness, the forensic normalization
code (repurposed as adversarial defense), and the measure-before-claiming
discipline that keeps a leaderboard-tuned model honest.
