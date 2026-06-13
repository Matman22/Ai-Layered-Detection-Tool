"""
Phase 3 - Learned weights via logistic regression (client-side ML).

Trains a logistic regression over all 18 detector signals on RAID, then folds
the StandardScaler back into raw-feature-space coefficients so the model ports
to JavaScript as a plain `sigmoid(sum(w_i * signal_i) + b)` - no scaler, no
model file, instant in-browser inference.

Compares the learned model head-to-head against the current hand-tuned WEIGHTS
on the SAME held-out test set. Reports threshold-independent ROC-AUC alongside
F1 so the comparison isn't gamed by threshold choice.

Usage:
    python optimize_weights.py                 # 2000 samples, abstracts
    python optimize_weights.py --n 3000
    python optimize_weights.py --penalty l1    # sparse weights (more interpretable)
    python optimize_weights.py --export weights.json
"""

import argparse
import json
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score, accuracy_score,
)

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets scikit-learn")
    sys.exit(1)

import detector as det

# Feature order — this is the contract with the JS port. Do not reorder.
FEATURE_NAMES = [
    "perplexity_proxy", "burstiness", "lexical_diversity", "ai_phrases",
    "hedging", "passive_voice", "transitions", "punctuation_variance",
    "paragraph_uniformity", "rare_words", "formality_shift", "ngram_repetition",
    "sentence_opener_diversity", "punctuation_fingerprint", "vocab_clustering",
    "density_melody", "forensic", "authorial",
]


def extract_features(text: str):
    """Returns (feature_vector[18], current_model_combined_score)."""
    pv = det.calc_punctuation_variance(text)
    sig = {
        "perplexity_proxy": det.calc_perplexity_proxy(text),
        "burstiness": det.calc_burstiness(text),
        "lexical_diversity": det.calc_lexical_diversity(text),
        "ai_phrases": det.calc_ai_phrases(text),
        "hedging": det.calc_hedging(text),
        "passive_voice": det.calc_passive_voice(text),
        "transitions": det.calc_transitions(text),
        "punctuation_variance": pv,
        "paragraph_uniformity": det.calc_paragraph_uniformity(text),
        "rare_words": det.calc_rare_words(text),
        "formality_shift": det.calc_formality_shift(text),
        "ngram_repetition": det.calc_ngram_repetition(text),
        "sentence_opener_diversity": det.calc_sentence_opener_diversity(text),
        "punctuation_fingerprint": det.calc_punctuation_fingerprint(text),
        "vocab_clustering": det.calc_vocab_clustering(text),
        "density_melody": det.calc_density_melody(text),
        "forensic": det.calc_forensic_score(text),
        "authorial": det.calc_authorial_score(text),
    }
    feat = np.array([sig[n] for n in FEATURE_NAMES], dtype=float)

    # Reproduce the current hand-tuned model's combined score from the same signals.
    slots = [
        sig["perplexity_proxy"], sig["burstiness"], sig["lexical_diversity"],
        sig["ai_phrases"], sig["hedging"], sig["passive_voice"], sig["transitions"],
        pv,  # clause-depth slot (proxy)
        pv,  # punctuation variance
        sig["paragraph_uniformity"], sig["rare_words"], sig["formality_shift"],
        sig["ngram_repetition"], sig["sentence_opener_diversity"],
        sig["punctuation_fingerprint"], sig["vocab_clustering"],
        sig["density_melody"], 0.0,
    ]
    slots[-1] = sum(slots[:-1]) / (len(slots) - 1)
    l1 = sum(w * s for w, s in zip(det.WEIGHTS, slots))
    combined = l1 * 0.75 + sig["forensic"] * 0.15 + sig["authorial"] * 0.10
    return feat, combined


def load_samples(n, domain, seed):
    random.seed(seed)
    print(f"Streaming RAID (domain={domain or 'all'})...")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    samples = []
    for row in ds:
        if domain and row.get("domain") != domain:
            continue
        if row.get("attack", "none") not in (None, "none", ""):
            continue  # clean (non-adversarial) only, to match the current benchmark
        samples.append({
            "text": row.get("generation", ""),
            "label": 0 if row.get("model") == "human" else 1,
        })
        if len(samples) >= n * 4:
            break
    random.shuffle(samples)
    return samples[:n]


def report(name, y_true, y_pred, y_score):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score),
    }


def print_row(r):
    print(f"  {r['model']:<22}  acc={r['accuracy']:.3f}  P={r['precision']:.3f}  "
          f"R={r['recall']:.3f}  F1={r['f1']:.4f}  AUC={r['auc']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--domain", type=str, default="abstracts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--penalty", choices=["l2", "l1"], default="l2")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--export", type=str, default="learned_weights.json")
    args = ap.parse_args()

    samples = load_samples(args.n, args.domain, args.seed)
    if len(samples) < 100:
        print(f"Only {len(samples)} samples loaded — need more.")
        sys.exit(1)

    print(f"Extracting features from {len(samples)} samples...")
    X, y, cur_score = [], [], []
    for s in samples:
        if len(s["text"].split()) < 20:
            continue
        feat, combined = extract_features(s["text"])
        X.append(feat)
        y.append(s["label"])
        cur_score.append(combined)
    X = np.array(X)
    y = np.array(y)
    cur_score = np.array(cur_score)
    print(f"Feature matrix: {X.shape}  (AI={int(y.sum())}, Human={int((y == 0).sum())})")

    # Held-out test split, stratified, same seed for reproducibility.
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.30, stratify=y, random_state=args.seed)

    # --- Baseline: current hand-tuned model on the test set ---
    # Current model predicts AI when combined >= 40 (Mixed counts as AI).
    base_pred = (cur_score[te] >= 40).astype(int)
    base = report("hand-tuned (current)", y[te], base_pred, cur_score[te])

    # --- Learned model: logistic regression (scaled) ---
    solver = "liblinear" if args.penalty == "l1" else "lbfgs"
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(penalty=args.penalty, C=args.C, solver=solver,
                                  class_weight="balanced", max_iter=2000)),
    ])
    pipe.fit(X[tr], y[tr])
    prob_te = pipe.predict_proba(X[te])[:, 1]
    lr_pred = (prob_te >= 0.5).astype(int)
    learned = report("learned (logreg)", y[te], lr_pred, prob_te)

    # 5-fold CV AUC on the full set for stability
    cv = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
                         scoring="roc_auc")

    print("\n" + "=" * 72)
    print(f"HEAD-TO-HEAD on held-out test set  (n_test={len(te)})")
    print("=" * 72)
    print_row(base)
    print_row(learned)
    print(f"\n  Learned model 5-fold CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}")
    d_f1 = learned["f1"] - base["f1"]
    d_auc = learned["auc"] - base["auc"]
    print(f"  Delta F1 : {d_f1:+.4f}    Delta AUC: {d_auc:+.4f}")
    verdict = "LEARNED WINS" if (d_auc > 0.005 or d_f1 > 0.005) else "NO MEANINGFUL GAIN"
    print(f"  >>> {verdict}")

    # --- Fold scaler into raw-feature-space weights for the JS port ---
    scaler = pipe.named_steps["scale"]
    lr = pipe.named_steps["lr"]
    coef = lr.coef_[0]
    intercept = lr.intercept_[0]
    raw_w = coef / scaler.scale_
    raw_b = intercept - np.sum(coef * scaler.mean_ / scaler.scale_)

    # Sanity check: raw-space logit must match the pipeline's logit.
    logit_pipe = pipe.decision_function(X[te])
    logit_raw = X[te] @ raw_w + raw_b
    assert np.allclose(logit_pipe, logit_raw, atol=1e-6), "fold-back mismatch!"

    print("\n" + "=" * 72)
    print("LEARNED FEATURE IMPORTANCE (standardized coef — sorted by magnitude)")
    print("=" * 72)
    order = np.argsort(-np.abs(coef))
    for i in order:
        bar = "#" * int(round(abs(coef[i]) / (abs(coef).max() + 1e-9) * 30))
        sign = "+AI" if coef[i] > 0 else "-AI"
        print(f"  {FEATURE_NAMES[i]:<28} {coef[i]:+.3f} [{sign}] {bar}")

    export = {
        "feature_names": FEATURE_NAMES,
        "raw_weights": [round(float(w), 6) for w in raw_w],
        "raw_intercept": round(float(raw_b), 6),
        "note": "score = 100 / (1 + exp(-(sum(raw_weights[i]*signal[i]) + raw_intercept)))",
        "test_f1": round(learned["f1"], 4),
        "test_auc": round(learned["auc"], 4),
        "cv_auc_mean": round(float(cv.mean()), 4),
    }
    with open(args.export, "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nExported raw-space weights to {args.export}")

    # Paste-ready JS snippet
    print("\n--- JS port (raw-feature-space) ---")
    print("const ML_WEIGHTS = [")
    for n, w in zip(FEATURE_NAMES, raw_w):
        print(f"  {w:+.6f}, // {n}")
    print("];")
    print(f"const ML_INTERCEPT = {raw_b:+.6f};")


if __name__ == "__main__":
    main()
