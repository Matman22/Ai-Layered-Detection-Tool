"""
Phase 3b - retune verdict thresholds for the learned (logistic) score.

The ML score is a calibrated P(AI)*100 spanning the full 0-100 range, so the
legacy bands (AI>=60 / Mixed>=40 / Human<40) that were built for the old
cramped 25-60 linear score no longer match it.

Only the LOWER cutoff (Human|Mixed boundary) affects binary AI/not-AI metrics,
since Mixed counts as AI. This sweeps that cutoff on a class-balanced RAID set
(so F1 isn't distorted by RAID's ~78% AI skew) and reports the best operating
points. Run live_eval.py afterwards to validate on the natural distribution.

Usage:
    python tune_thresholds.py            # ~1200 balanced abstracts
    python tune_thresholds.py --n 1600
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets"); sys.exit(1)
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
RUNNER = SCRIPT_DIR / "node_runner.js"


def load_balanced(n, domain, seed, max_window=60000):
    random.seed(seed)
    target = n // 2
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    human, ai, seen = [], [], 0
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
                human.append({"id": 0, "text": text, "label": 0})
        elif len(ai) < target:
            ai.append({"id": 0, "text": text, "label": 1})
        if len(human) >= target and len(ai) >= target:
            break
    samples = human + ai
    for i, s in enumerate(samples):
        s["id"] = i
    random.shuffle(samples)
    print(f"Balanced set: {len(human)} human + {len(ai)} AI")
    return samples


def score_samples(samples, node_cmd, batch=100):
    out = {}
    batches = [samples[i:i+batch] for i in range(0, len(samples), batch)]
    for b in tqdm(batches, desc="Scoring (node)"):
        payload = json.dumps([{"id": s["id"], "text": s["text"]} for s in b])
        proc = subprocess.run([node_cmd, str(RUNNER)], input=payload.encode("utf-8"),
                              capture_output=True, timeout=180)
        if proc.returncode != 0:
            print(proc.stderr.decode()[:200]); continue
        for r in json.loads(proc.stdout.decode("utf-8")):
            out[r["id"]] = r["score"]
    return out


def metrics_at(scores, labels, t):
    tp = fp = tn = fn = 0
    for sc, y in zip(scores, labels):
        pred = 1 if sc >= t else 0
        if y == 1 and pred == 1: tp += 1
        elif y == 0 and pred == 1: fp += 1
        elif y == 0 and pred == 0: tn += 1
        else: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(labels)
    return {"t": t, "P": prec, "R": rec, "F1": f1, "acc": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--domain", type=str, default="abstracts")
    ap.add_argument("--seed", type=int, default=7)  # different seed than training
    args = ap.parse_args()

    node_cmd = "node"
    try:
        subprocess.run([node_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        fb = r"C:\Program Files\nodejs\node.exe"
        node_cmd = fb if Path(fb).exists() else node_cmd

    samples = load_balanced(args.n, args.domain, args.seed)
    sc = score_samples(samples, node_cmd)
    scores = [sc[s["id"]] for s in samples if s["id"] in sc]
    labels = [s["label"] for s in samples if s["id"] in sc]
    print(f"Scored {len(scores)} samples. Score range: {min(scores)}-{max(scores)}\n")

    print("=" * 60)
    print("THRESHOLD SWEEP (lower cutoff = binary AI boundary)")
    print("=" * 60)
    print(f"  {'t':>3}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Acc':>6}   TP/FP/TN/FN")
    rows = [metrics_at(scores, labels, t) for t in range(20, 76, 5)]
    for r in rows:
        print(f"  {r['t']:>3}  {r['P']:.3f}  {r['R']:.3f}  {r['F1']:.3f}  {r['acc']:.3f}   "
              f"{r['tp']}/{r['fp']}/{r['tn']}/{r['fn']}")

    # finer sweep around best F1
    best_f1 = max(rows, key=lambda r: r["F1"])
    fine = [metrics_at(scores, labels, t) for t in range(max(20, best_f1["t"] - 4), best_f1["t"] + 5)]
    best_f1 = max(fine, key=lambda r: r["F1"])

    # precision-favoring option: max recall subject to precision >= 0.95
    hp = [metrics_at(scores, labels, t) for t in range(20, 81)]
    prec_constrained = [r for r in hp if r["P"] >= 0.95]
    best_hp = max(prec_constrained, key=lambda r: r["R"]) if prec_constrained else None

    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"  Best F1:           t={best_f1['t']}  F1={best_f1['F1']:.3f}  "
          f"P={best_f1['P']:.3f}  R={best_f1['R']:.3f}")
    if best_hp:
        print(f"  Max recall @ P>=0.95: t={best_hp['t']}  F1={best_hp['F1']:.3f}  "
              f"P={best_hp['P']:.3f}  R={best_hp['R']:.3f}")
    print("\n  Lower cutoff = Human|Mixed boundary (binary AI threshold).")
    print("  Pick an upper cutoff (Mixed|AI) ~15-20 pts above for a confident-AI band.")


if __name__ == "__main__":
    main()
