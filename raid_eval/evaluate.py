"""
RAID benchmark evaluation for the AI Layered Detection Tool Python port.

Usage:
    python evaluate.py                          # quick run: 500 samples, all models
    python evaluate.py --n 2000                 # larger sample
    python evaluate.py --domain news            # filter by domain
    python evaluate.py --model gpt4             # filter by model
    python evaluate.py --adversarial            # include adversarial splits only
    python evaluate.py --output results.json    # save full results

Requires: pip install raid-bench datasets scikit-learn tqdm
"""

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

RAID_AVAILABLE = False  # raid-bench requires a C compiler; we use HuggingFace directly

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from tqdm import tqdm

from detector import score_text

try:  # Windows consoles default to cp1252 and choke on unicode glyphs in output
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ---------------------------------------------------------------------------
# RAID data loading
# ---------------------------------------------------------------------------

def load_raid_data(n: int, domain: str | None, model: str | None, adversarial: bool):
    """
    Load labeled samples from RAID. Falls back to HuggingFace datasets if
    the raid-bench package loader isn't available.
    Returns list of dicts: {text, label (0=human/1=AI), model, domain, attack}
    """
    samples = []

    if RAID_AVAILABLE:
        print("Loading via raid-bench package...")
        try:
            data = load_data(split="train")
            for row in data:
                if domain and row.get("domain") != domain:
                    continue
                if model and row.get("model") != model:
                    continue
                if adversarial and row.get("attack") in (None, "none", ""):
                    continue
                samples.append({
                    "text": row["generation"],
                    "label": 0 if row.get("label") in ("human", 0) else 1,
                    "model": row.get("model", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "attack": row.get("attack", "none"),
                })
        except Exception as e:
            print(f"raid-bench loader failed ({e}), trying HuggingFace...")
            RAID_AVAILABLE_LOCAL = False
        else:
            RAID_AVAILABLE_LOCAL = True
    else:
        RAID_AVAILABLE_LOCAL = False

    if not RAID_AVAILABLE_LOCAL:
        if not HF_AVAILABLE:
            print("ERROR: Install either raid-bench or datasets (HuggingFace).")
            print("  pip install raid-bench datasets scikit-learn tqdm")
            sys.exit(1)
        print("Loading via HuggingFace datasets (ldugan/raid)...")
        ds = load_dataset("liamdugan/raid", split="train", streaming=True)
        for row in ds:
            row_model = row.get("model", "unknown")
            row_domain = row.get("domain", "unknown")
            row_attack = row.get("attack", "none")
            if domain and row_domain != domain:
                continue
            if model and row_model != model:
                continue
            if adversarial and row_attack in (None, "none", ""):
                continue
            samples.append({
                "text": row.get("generation", row.get("text", "")),
                "label": 0 if row_model == "human" else 1,
                "model": row_model,
                "domain": row_domain,
                "attack": row_attack,
            })
            if len(samples) >= n * 3:  # over-sample then trim
                break

    # Shuffle and cap
    random.shuffle(samples)
    return samples[:n]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(samples: list[dict]) -> dict:
    y_true, y_pred, scores, per_model, per_domain, per_attack = [], [], [], defaultdict(list), defaultdict(list), defaultdict(list)

    print(f"\nRunning detector on {len(samples)} samples...")
    t0 = time.time()

    for s in tqdm(samples):
        result = score_text(s["text"])
        pred = result["prediction"]
        true = s["label"]

        y_true.append(true)
        y_pred.append(pred)
        scores.append(result["combined_score"])

        per_model[s["model"]].append((true, pred))
        per_domain[s["domain"]].append((true, pred))
        per_attack[s["attack"]].append((true, pred))

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / len(samples) * 1000:.1f}ms/sample)")

    def metrics(yt, yp):
        if len(set(yt)) < 2:
            return {"note": "only one class present"}
        return {
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall":    round(recall_score(yt, yp, zero_division=0), 4),
            "f1":        round(f1_score(yt, yp, zero_division=0), 4),
            "accuracy":  round(sum(a == b for a, b in zip(yt, yp)) / len(yt), 4),
            "n":         len(yt),
        }

    def breakdown(group_dict):
        return {k: metrics([a for a, _ in v], [b for _, b in v]) for k, v in sorted(group_dict.items())}

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

    return {
        "overall": metrics(y_true, y_pred),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "score_distribution": {
            "mean": round(sum(scores) / len(scores), 2),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
        },
        "by_model": breakdown(per_model),
        "by_domain": breakdown(per_domain),
        "by_attack": breakdown(per_attack),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_results(results: dict):
    ov = results["overall"]
    cm = results["confusion_matrix"]

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Samples   : {ov['n']}")
    print(f"  Accuracy  : {ov['accuracy']:.1%}")
    print(f"  Precision : {ov['precision']:.1%}  (of flagged AI, how many actually are)")
    print(f"  Recall    : {ov['recall']:.1%}  (of actual AI, how many we caught)")
    print(f"  F1        : {ov['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives  (AI->AI)    : {cm['TP']}")
    print(f"  False Positives (Human->AI) : {cm['FP']}")
    print(f"  True Negatives  (Human->H.) : {cm['TN']}")
    print(f"  False Negatives (AI->Human) : {cm['FN']}")

    sd = results["score_distribution"]
    print(f"\nScore distribution  mean={sd['mean']}  min={sd['min']}  max={sd['max']}")

    def print_breakdown(title, data):
        print(f"\n{title}")
        print("-" * 60)
        for key, m in data.items():
            if "note" in m:
                print(f"  {key:30s}  (single-class, skipped)")
            else:
                print(f"  {key:30s}  acc={m['accuracy']:.1%}  f1={m['f1']:.3f}  n={m['n']}")

    print_breakdown("BY MODEL", results["by_model"])
    print_breakdown("BY DOMAIN", results["by_domain"])
    print_breakdown("BY ATTACK TYPE", results["by_attack"])
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Multi-dataset evaluation (cross-source generalization)
# ---------------------------------------------------------------------------

import csv as _csv
from pathlib import Path as _Path

_csv.field_size_limit(10_000_000)  # essays/abstracts can be long

EVAL_DIR = _Path(__file__).parent.parent / "eval"
MULTI_CSV = EVAL_DIR / "multi_dataset.csv"
ADV_CSV = EVAL_DIR / "adversarial_raid.csv"


def read_multi_csv(path):
    """Read the unified multi-dataset CSV into row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in _csv.DictReader(f):
            text = (r.get("text_content") or "").strip()
            if len(text.split()) < 20:
                continue
            rows.append({
                "text": text,
                "label": int(r["is_ai_generated"]),
                "source": r.get("source_dataset", "") or "",
                "domain": r.get("domain", "") or "",
                "model": r.get("model", "") or "",
            })
    return rows


def _prf(y_true, y_pred):
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "acc": round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 4),
        "n": len(y_true),
        "n_ai": sum(y_true),
    }


def score_rows(rows):
    print(f"Scoring {len(rows)} rows through detector.score_text...")
    for r in tqdm(rows):
        r["pred"] = score_text(r["text"])["prediction"]
    return rows


def _line(label, m):
    if m["n_ai"] in (0, m["n"]):
        tag = "AI-only" if m["n_ai"] == m["n"] else "human-only"
        print(f"  {label:<22} ({tag}, n={m['n']})  recall={m['recall']:.3f}  prec={m['precision']:.3f}")
    else:
        print(f"  {label:<22} P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
              f"acc={m['acc']:.3f}  n={m['n']}")


def report_per_source(rows):
    print("\n" + "=" * 64)
    print("PER-SOURCE  (rule-based detector — fixed, no training)")
    print("=" * 64)
    by = defaultdict(lambda: {"t": [], "p": []})
    for r in rows:
        by[r["source"]]["t"].append(r["label"])
        by[r["source"]]["p"].append(r["pred"])
    for src in sorted(by):
        _line(src, _prf(by[src]["t"], by[src]["p"]))


def report_cross_dataset(rows):
    # For a fixed rule-based detector there is no train step, so the cross-dataset
    # matrix collapses to the per-source diagonal. The real train-A/test-B matrix
    # is produced by train_stylometric.py once a learned classifier exists.
    print("\n" + "=" * 64)
    print("CROSS-DATASET  (rule-based: diagonal only — see train_stylometric.py")
    print("                for the real train-A / test-B generalization matrix)")
    print("=" * 64)
    by = defaultdict(lambda: {"t": [], "p": []})
    for r in rows:
        by[r["source"]]["t"].append(r["label"])
        by[r["source"]]["p"].append(r["pred"])
    srcs = sorted(by)
    print("  test→".ljust(24) + "  ".join(s[:10].rjust(10) for s in srcs))
    print(f"  {'(fixed detector)':<22}" + "  ".join(
        f"{_prf(by[s]['t'], by[s]['p'])['f1']:.3f}".rjust(10) for s in srcs))


def report_leave_one_model_out(rows):
    print("\n" + "=" * 64)
    print("LEAVE-ONE-MODEL-OUT  (F1 on each held-out generator family)")
    print("=" * 64)
    by_model = defaultdict(lambda: {"t": [], "p": []})
    for r in rows:
        if r["model"] and r["label"] == 1:  # generator families are AI rows
            by_model[r["model"]]["t"].append(r["label"])
            by_model[r["model"]]["p"].append(r["pred"])
    if not by_model:
        print("  (no model labels present)")
        return
    for m in sorted(by_model, key=lambda k: -len(by_model[k]["t"])):
        d = by_model[m]
        # recall = fraction of this model's AI text we caught (single-class slice)
        rec = sum(d["p"]) / len(d["p"])
        print(f"  {m:<26} recall={rec:.3f}  (n={len(d['p'])} AI samples)")


def report_adversarial():
    if not ADV_CSV.exists():
        print("\n(no adversarial_raid.csv — skipping adversarial slice)")
        return
    rows = score_rows(read_multi_csv(ADV_CSV))
    print("\n" + "=" * 64)
    print("ADVERSARIAL SLICE  (held-out RAID attack!=\"none\")")
    print("=" * 64)
    _line("raid_adversarial", _prf([r["label"] for r in rows], [r["pred"] for r in rows]))


def multi_eval():
    if not MULTI_CSV.exists():
        print(f"Missing {MULTI_CSV}. Run: node eval/fetch_dataset.js --multi --per-source 400 --balance")
        sys.exit(1)
    rows = score_rows(read_multi_csv(MULTI_CSV))
    overall = _prf([r["label"] for r in rows], [r["pred"] for r in rows])
    print("\n" + "=" * 64)
    print(f"OVERALL (all sources pooled)  n={overall['n']}")
    print("=" * 64)
    _line("ALL", overall)
    report_per_source(rows)
    report_cross_dataset(rows)
    report_leave_one_model_out(rows)
    report_adversarial()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Detector against RAID benchmark")
    parser.add_argument("--multi", action="store_true",
                        help="Evaluate against eval/multi_dataset.csv (per-source, cross-dataset, "
                             "leave-one-model-out, adversarial)")
    parser.add_argument("--n", type=int, default=500, help="Number of samples (default 500)")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain (e.g. news, reddit)")
    parser.add_argument("--model", type=str, default=None, help="Filter by LLM (e.g. gpt4, llama2)")
    parser.add_argument("--adversarial", action="store_true", help="Include only adversarial attack samples")
    parser.add_argument("--output", type=str, default=None, help="Save full results JSON to this path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.multi:
        multi_eval()
        return

    samples = load_raid_data(args.n, args.domain, args.model, args.adversarial)
    if not samples:
        print("No samples loaded — check your filters or install dependencies.")
        sys.exit(1)

    print(f"Loaded {len(samples)} samples  "
          f"(AI: {sum(s['label'] for s in samples)}, "
          f"Human: {sum(1 - s['label'] for s in samples)})")

    results = evaluate(samples)
    print_results(results)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    main()
