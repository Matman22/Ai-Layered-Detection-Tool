"""
Phase 3 (client-side ML) - train learned weights on the ACTUAL JS signals.

Why this exists: the live site computes its 20 signals in JavaScript. Their
value distributions differ from the Python port (esp. perplexity + authorial),
so a model trained on Python features would not transfer. This script extracts
features via node_runner_verbose.js (the real index.html logic), trains a
logistic regression on a class-balanced RAID set, and exports raw-feature-space
weights that drop straight into index.html as sigmoid(sum(w*signal)+b).

Honesty guards:
  - class-balanced collection (RAID is ~87% AI; balancing makes F1 meaningful)
  - held-out test split + 5-fold CV, both reported
  - ROC-AUC (threshold-free) is the headline metric
  - head-to-head vs the current hand-tuned JS model on the same test set

Usage:
    python optimize_weights_js.py                 # ~1500 balanced abstracts
    python optimize_weights_js.py --n 2000 --penalty l1
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets scikit-learn")
    sys.exit(1)

SCRIPT_DIR = Path(__file__).parent
VERBOSE_RUNNER = SCRIPT_DIR / "node_runner_verbose.js"

# JS feature order — the contract with the index.html port. Do not reorder.
FEATURE_NAMES = [
    "perplexity", "burstiness", "lexical_diversity", "ai_phrases", "hedging",
    "passive_voice", "transitions", "clause_depth", "punctuation_variance",
    "paragraph_uniformity", "rare_words", "formality_shift", "ngram_repetition",
    "sentence_opener_diversity", "punctuation_fingerprint", "vocab_clustering",
    "density_melody", "monte_carlo", "forensic", "authorial",
]

# Current hand-tuned JS weights (index.html 18-slot array) for the baseline.
HAND_WEIGHTS = [0.00, 0.11, 0.13, 0.07, 0.00, 0.00, 0.00, 0.02, 0.07, 0.12,
                0.05, 0.12, 0.04, 0.08, 0.04, 0.07, 0.03, 0.05]


def load_balanced(n, domain, seed, max_window=60000):
    """Collect ~n/2 human + ~n/2 AI clean abstracts by streaming RAID."""
    import random
    random.seed(seed)
    target = n // 2
    print(f"Streaming RAID for a balanced set (target {target}/class)...")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    human, ai = [], []
    seen = 0
    for row in ds:
        seen += 1
        if seen > max_window:
            break
        if domain and row.get("domain") != domain:
            continue
        if row.get("attack", "none") not in (None, "none", ""):
            continue
        text = row.get("generation", "")
        if len(text.split()) < 20:
            continue
        if row.get("model") == "human":
            if len(human) < target:
                human.append({"text": text, "label": 0})
        else:
            if len(ai) < target:
                ai.append({"text": text, "label": 1})
        if len(human) >= target and len(ai) >= target:
            break
    samples = human + ai
    random.shuffle(samples)
    print(f"Collected {len(human)} human + {len(ai)} AI = {len(samples)} (streamed {seen} rows)")
    return samples


def extract_js_features(samples, node_cmd, batch_size=100):
    """Pipe texts through node_runner_verbose.js, return feature matrix + ids kept."""
    from tqdm import tqdm
    feats, labels = [], []
    items = [{"id": i, "text": s["text"]} for i, s in enumerate(samples)]
    batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
    results = {}
    for batch in tqdm(batches, desc="JS feature extraction"):
        payload = json.dumps(batch)
        proc = subprocess.run([node_cmd, str(VERBOSE_RUNNER)],
                              input=payload.encode("utf-8"),
                              capture_output=True, timeout=180)
        if proc.returncode != 0:
            print(proc.stderr.decode()[:300]); continue
        for r in json.loads(proc.stdout.decode("utf-8")):
            results[r["id"]] = r

    base_scores = []
    for i, s in enumerate(samples):
        r = results.get(i)
        if not r or not r.get("signals"):
            continue
        sig = r["signals"]
        row = [
            sig["perplexity"], sig["burstiness"], sig["lexical_diversity"],
            sig["ai_phrases"], sig["hedging"], sig["passive_voice"], sig["transitions"],
            sig["clause_depth"], sig["punctuation_variance"], sig["paragraph_uniformity"],
            sig["rare_words"], sig["formality_shift"], sig["ngram_repetition"],
            sig["sentence_opener_diversity"], sig["punctuation_fingerprint"],
            sig["vocab_clustering"], sig["density_melody"], sig["monte_carlo"],
            r["l2"], r["l5"],
        ]
        feats.append(row)
        labels.append(s["label"])
        # Reproduce current hand-tuned JS combined score from the same signals.
        slots = row[:17] + [0.0]
        slots[17] = sum(slots[:17]) / 17
        composite = sum(w * v for w, v in zip(HAND_WEIGHTS, slots))
        combined = composite * 0.75 + r["l2"] * 0.15 + r["l5"] * 0.10
        base_scores.append(combined)
    return np.array(feats), np.array(labels), np.array(base_scores)


def report(name, y, pred, score):
    return {"model": name, "acc": accuracy_score(y, pred),
            "P": precision_score(y, pred, zero_division=0),
            "R": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "auc": roc_auc_score(y, score)}


def row(r):
    print(f"  {r['model']:<22} acc={r['acc']:.3f}  P={r['P']:.3f}  R={r['R']:.3f}  "
          f"F1={r['f1']:.4f}  AUC={r['auc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--domain", type=str, default="abstracts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--penalty", choices=["l2", "l1"], default="l2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--export", type=str, default="learned_weights_js.json")
    args = ap.parse_args()

    node_cmd = "node"
    try:
        subprocess.run([node_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        fb = r"C:\Program Files\nodejs\node.exe"
        if Path(fb).exists():
            node_cmd = fb
        else:
            print("ERROR: node not found."); sys.exit(1)

    samples = load_balanced(args.n, args.domain, args.seed)
    if len(samples) < 200:
        print("Not enough samples."); sys.exit(1)

    X, y, base_score = extract_js_features(samples, node_cmd)
    print(f"Feature matrix: {X.shape}  (AI={int(y.sum())}, Human={int((y==0).sum())})")

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)

    # Baseline: current hand-tuned JS model (predict AI when combined >= 40).
    base = report("hand-tuned (current)", y[te], (base_score[te] >= 40).astype(int), base_score[te])

    solver = "liblinear" if args.penalty == "l1" else "lbfgs"
    pipe = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(penalty=args.penalty, C=args.C, solver=solver,
                                               class_weight="balanced", max_iter=2000))])
    pipe.fit(X[tr], y[tr])
    prob = pipe.predict_proba(X[te])[:, 1]
    learned = report("learned (logreg)", y[te], (prob >= 0.5).astype(int), prob)
    cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                         scoring="roc_auc")

    print("\n" + "=" * 72)
    print(f"HEAD-TO-HEAD on balanced held-out test set (n_test={len(te)})")
    print("=" * 72)
    row(base); row(learned)
    print(f"\n  Learned 5-fold CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}")
    print(f"  Delta F1: {learned['f1']-base['f1']:+.4f}   Delta AUC: {learned['auc']-base['auc']:+.4f}")
    print(f"  >>> {'LEARNED WINS' if (learned['auc']-base['auc'] > 0.005) else 'NO MEANINGFUL GAIN'}")

    # Fold scaler into raw JS-feature-space weights.
    sc, lr = pipe.named_steps["scale"], pipe.named_steps["lr"]
    coef, b0 = lr.coef_[0], lr.intercept_[0]
    raw_w = coef / sc.scale_
    raw_b = b0 - np.sum(coef * sc.mean_ / sc.scale_)
    assert np.allclose(pipe.decision_function(X[te]), X[te] @ raw_w + raw_b, atol=1e-6)

    print("\n" + "=" * 72)
    print("FEATURE IMPORTANCE (standardized coef, sorted)")
    print("=" * 72)
    for i in np.argsort(-np.abs(coef)):
        bar = "#" * int(round(abs(coef[i]) / (abs(coef).max() + 1e-9) * 28))
        print(f"  {FEATURE_NAMES[i]:<28} {coef[i]:+.3f} [{'+AI' if coef[i]>0 else '-AI'}] {bar}")

    out = {"feature_names": FEATURE_NAMES,
           "raw_weights": [round(float(w), 6) for w in raw_w],
           "raw_intercept": round(float(raw_b), 6),
           "scoring": "score = 100/(1+exp(-(sum(w[i]*signal[i])+b)))",
           "test_f1": round(learned["f1"], 4), "test_auc": round(learned["auc"], 4),
           "cv_auc_mean": round(float(cv.mean()), 4), "cv_auc_std": round(float(cv.std()), 4),
           "baseline_test_f1": round(base["f1"], 4), "baseline_test_auc": round(base["auc"], 4)}
    Path(args.export).write_text(json.dumps(out, indent=2))
    print(f"\nExported -> {args.export}")
    print("\n--- index.html port ---")
    print("const ML_WEIGHTS = [")
    for n_, w in zip(FEATURE_NAMES, raw_w):
        print(f"  {w:+.6f}, // {n_}")
    print("];")
    print(f"const ML_INTERCEPT = {raw_b:+.6f};")


if __name__ == "__main__":
    main()
