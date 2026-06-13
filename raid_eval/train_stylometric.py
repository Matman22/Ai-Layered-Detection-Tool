"""
Step 5 — stylometric classifier with generalization-focused evaluation.

Trains a logistic regression on eval/features.csv (the calc_* signal matrix) and
reports the metrics that actually prove generalization rather than corpus
memorization:

  - in-sample 5-fold CV F1            (reference / optimistic ceiling)
  - cross-dataset matrix              (train on source A, test on source B)
  - leave-one-model-out recall        (hold out a generator family entirely)

The headline number is the gap between in-sample F1 and mean cross-dataset F1.
Exports model.json = {feature_names, mean, std, weights, bias} (standardized-space
weights, so inference is: z=(x-mean)/std; p=sigmoid(sum(w*z)+bias)).

    python train_stylometric.py
    python train_stylometric.py --features ../eval/features.csv --export ../model.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:  # Windows consoles default to cp1252 and choke on unicode glyphs in output
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score, recall_score, confusion_matrix

csv.field_size_limit(10_000_000)
REPO = Path(__file__).parent.parent


def load_features(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    meta_cols = {"is_ai_generated", "source_dataset", "model"}
    feat_names = [c for c in header if c not in meta_cols]
    idx = {c: i for i, c in enumerate(header)}
    X, y, src, model = [], [], [], []
    for r in rows:
        X.append([float(r[idx[c]]) for c in feat_names])
        y.append(int(r[idx["is_ai_generated"]]))
        src.append(r[idx["source_dataset"]])
        model.append(r[idx["model"]])
    return feat_names, np.array(X), np.array(y), np.array(src), np.array(model)


def new_model():
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])


def fit_eval(Xtr, ytr, Xte, yte):
    if len(set(ytr)) < 2 or len(Xtr) < 20:
        return None
    m = new_model()
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    if len(set(yte)) < 2:
        return {"recall": recall_score(yte, pred, zero_division=0), "f1": None, "n": len(yte)}
    return {"f1": f1_score(yte, pred, zero_division=0),
            "recall": recall_score(yte, pred, zero_division=0), "n": len(yte)}


def cross_dataset_matrix(X, y, src, feat_names):
    sources = sorted(set(src))
    print("\n" + "=" * 70)
    print("CROSS-DATASET F1 MATRIX  (rows = train source, cols = test source)")
    print("=" * 70)
    print("  train↓ / test→".ljust(20) + "".join(s[:9].rjust(10) for s in sources))
    diag, offdiag = [], []
    for a in sources:
        cells = []
        for b in sources:
            tr = src == a
            te = src == b
            if a == b:
                # honest in-source number via CV (no train/test leakage)
                if len(set(y[tr])) < 2 or tr.sum() < 30:
                    cells.append("   n/a"); continue
                cv = cross_val_score(new_model(), X[tr], y[tr],
                                     cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                     scoring="f1")
                f1 = cv.mean(); diag.append(f1)
            else:
                r = fit_eval(X[tr], y[tr], X[te], y[te])
                if r is None or r["f1"] is None:
                    cells.append("   n/a"); continue
                f1 = r["f1"]; offdiag.append(f1)
            cells.append(f"{f1:.3f}".rjust(10))
        print(f"  {a[:18]:<18}" + "".join(cells))
    if diag and offdiag:
        print(f"\n  mean in-source (diagonal) F1 : {np.mean(diag):.3f}")
        print(f"  mean cross-source F1         : {np.mean(offdiag):.3f}")
        print(f"  >>> GENERALIZATION GAP        : {np.mean(diag) - np.mean(offdiag):+.3f}  "
              f"(smaller = generalizes better)")


def leave_one_model_out(X, y, src, model):
    print("\n" + "=" * 70)
    print("LEAVE-ONE-MODEL-OUT  (train without a generator family, test on it)")
    print("=" * 70)
    families = sorted({m for m, lbl in zip(model, y) if m and lbl == 1})
    families = [m for m in families if (model == m).sum() >= 10]
    if not families:
        print("  (no generator families with >=10 samples)")
        return
    recalls = []
    for m in families:
        held = model == m
        tr = ~held
        if len(set(y[tr])) < 2:
            continue
        clf = new_model()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[held])
        rec = recall_score(y[held], pred, zero_division=0)
        recalls.append(rec)
        print(f"  {m:<28} recall={rec:.3f}  (n={int(held.sum())} held-out AI samples)")
    if recalls:
        print(f"\n  mean held-out recall across {len(recalls)} families: {np.mean(recalls):.3f}")


def report_informal_holdout(X, y, src, informal="andy"):
    """Success-criterion check: beat the old single-dataset F1 (0.489) on held-out
    informal text without specificity dropping below ~0.798. Two views:
      (a) diverse training WITH informal in the mix, held-out informal test split
      (b) leave-one-dataset-out: informal source never seen in training (the
          honest cross-corpus ceiling)."""
    print("\n" + "=" * 70)
    print(f"HELD-OUT INFORMAL ('{informal}')  vs baseline F1=0.489 / spec floor=0.798")
    print("=" * 70)
    if informal not in set(src):
        print(f"  (no '{informal}' rows present)")
        return

    def spec_of(yt, p):
        tn, fp, fn, tp = confusion_matrix(yt, p, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) else 0.0

    # (a) diverse training incl. informal; evaluate on the informal test slice
    ii = np.arange(len(y))
    strat = [f"{s}{l}" for s, l in zip(src, y)]
    tr, te = train_test_split(ii, test_size=0.30, stratify=strat, random_state=42)
    m = new_model().fit(X[tr], y[tr])
    mask = np.array([i for i in te if src[i] == informal])
    p = m.predict(X[mask])
    f1a, spa = f1_score(y[mask], p), spec_of(y[mask], p)
    print(f"  (a) diverse train (incl. informal) -> held-out informal test (n={len(mask)})")
    print(f"        F1={f1a:.3f}  recall={recall_score(y[mask], p):.3f}  specificity={spa:.3f}"
          f"   {'PASS' if (f1a > 0.489 and spa >= 0.798) else 'FAIL'}")

    # (b) leave-one-dataset-out: informal source entirely unseen
    trb, teb = src != informal, src == informal
    if len(set(y[trb])) == 2:
        pb = new_model().fit(X[trb], y[trb]).predict(X[teb])
        print(f"  (b) leave-one-dataset-out ('{informal}' unseen in training, n={int(teb.sum())})")
        print(f"        F1={f1_score(y[teb], pb):.3f}  recall={recall_score(y[teb], pb):.3f}  "
              f"specificity={spec_of(y[teb], pb):.3f}   <- cross-corpus ceiling")


def export_model(X, y, feat_names, path):
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000)
    lr.fit(scaler.transform(X), y)
    payload = {
        "feature_names": feat_names,
        "mean": [round(v, 6) for v in scaler.mean_.tolist()],
        "std": [round(v, 6) for v in scaler.scale_.tolist()],
        "weights": [round(v, 6) for v in lr.coef_[0].tolist()],
        "bias": round(float(lr.intercept_[0]), 6),
        "scoring": "z=(x-mean)/std; p=1/(1+exp(-(sum(weights*z)+bias)))",
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"\nExported trained model -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(REPO / "eval" / "features.csv"))
    ap.add_argument("--export", default=str(REPO / "model.json"))
    args = ap.parse_args()

    if not Path(args.features).exists():
        print(f"Missing {args.features}. Run: python extract_features.py")
        return

    feat_names, X, y, src, model = load_features(args.features)
    print(f"Loaded {len(y)} rows x {len(feat_names)} features  "
          f"(AI={int(y.sum())}, Human={int((y == 0).sum())}, sources={sorted(set(src))})")

    # In-sample reference: pooled 5-fold CV (optimistic — same-corpus train/test)
    cv = cross_val_score(new_model(), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
    print("\n" + "=" * 70)
    print(f"IN-SAMPLE pooled 5-fold CV F1: {cv.mean():.3f} +/- {cv.std():.3f}  (optimistic ceiling)")
    print("=" * 70)

    cross_dataset_matrix(X, y, src, feat_names)
    leave_one_model_out(X, y, src, model)
    report_informal_holdout(X, y, src)
    export_model(X, y, feat_names, args.export)


if __name__ == "__main__":
    main()
