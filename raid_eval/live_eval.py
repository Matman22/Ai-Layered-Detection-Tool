"""
Live-site evaluation: runs RAID samples through the actual JS detection
logic extracted from index.html via Node.js.

Usage:
    python live_eval.py                         # 200 samples, all domains
    python live_eval.py --n 500
    python live_eval.py --domain news
    python live_eval.py --adversarial
    python live_eval.py --model gpt4
    python live_eval.py --compare              # also run Python port side-by-side
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets scikit-learn tqdm")
    sys.exit(1)

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
NODE_RUNNER = SCRIPT_DIR / "node_runner.js"


# ─── Data loading ────────────────────────────────────────────────────────────

def load_samples(n, domain, model, adversarial, seed):
    random.seed(seed)
    print("Streaming RAID from HuggingFace...")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    samples = []
    for row in ds:
        rm  = row.get("model", "unknown")
        rd  = row.get("domain", "unknown")
        ra  = row.get("attack", "none")
        if domain     and rd != domain: continue
        if model      and rm != model:  continue
        if adversarial and ra in (None, "none", ""): continue
        samples.append({
            "id":     len(samples),
            "text":   row.get("generation", ""),
            "label":  0 if rm == "human" else 1,
            "model":  rm,
            "domain": rd,
            "attack": ra,
        })
        if len(samples) >= n * 4:
            break
    random.shuffle(samples)
    return samples[:n]


# ─── Node.js batch runner ────────────────────────────────────────────────────

def run_node(samples, node_cmd="node", batch_size=100):
    """Pipe samples through node_runner.js in batches. Returns {id: result}."""
    all_results = {}
    batches = [samples[i:i+batch_size] for i in range(0, len(samples), batch_size)]

    for batch in tqdm(batches, desc="Node.js batches"):
        payload = json.dumps([{"id": s["id"], "text": s["text"]} for s in batch])
        try:
            proc = subprocess.run(
                [node_cmd, str(NODE_RUNNER)],
                input=payload.encode("utf-8"),
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            print("  WARNING: batch timed out, skipping")
            continue

        if proc.returncode != 0:
            print(f"  Node stderr: {proc.stderr.decode()[:300]}")
            continue

        stderr = proc.stderr.decode().strip()
        if stderr:
            # Print non-fatal warnings but don't fail
            for line in stderr.splitlines()[:3]:
                if "non-fatal" not in line:
                    print(f"  [node] {line}")

        try:
            batch_results = json.loads(proc.stdout.decode("utf-8"))
            for r in batch_results:
                all_results[r["id"]] = r
        except json.JSONDecodeError as e:
            print(f"  JSON decode error: {e}")

    return all_results


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    if len(set(y_true)) < 2:
        return {"note": "single-class only"}
    return {
        "accuracy":  round(sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n":         len(y_true),
    }


def evaluate(samples, node_results):
    y_true, y_pred = [], []
    per_model  = defaultdict(lambda: {"true": [], "pred": []})
    per_domain = defaultdict(lambda: {"true": [], "pred": []})
    per_attack = defaultdict(lambda: {"true": [], "pred": []})
    scores = []

    for s in samples:
        r = node_results.get(s["id"])
        if r is None or r.get("verdict") == "Error":
            continue
        # Verdict mapping: AI or Mixed → 1, Human → 0
        pred = 0 if r["verdict"] == "Human" else 1
        true = s["label"]
        y_true.append(true)
        y_pred.append(pred)
        scores.append(r["score"])
        per_model[s["model"]]["true"].append(true);  per_model[s["model"]]["pred"].append(pred)
        per_domain[s["domain"]]["true"].append(true); per_domain[s["domain"]]["pred"].append(pred)
        per_attack[s["attack"]]["true"].append(true); per_attack[s["attack"]]["pred"].append(pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "overall":    compute_metrics(y_true, y_pred),
        "confusion":  {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        "scores":     {"mean": round(sum(scores)/len(scores),1), "min": round(min(scores),1), "max": round(max(scores),1)},
        "by_model":   {k: compute_metrics(v["true"], v["pred"]) for k, v in sorted(per_model.items())},
        "by_domain":  {k: compute_metrics(v["true"], v["pred"]) for k, v in sorted(per_domain.items())},
        "by_attack":  {k: compute_metrics(v["true"], v["pred"]) for k, v in sorted(per_attack.items())},
    }


# ─── Pretty print ────────────────────────────────────────────────────────────

def print_results(results, title="LIVE SITE (JS) RESULTS"):
    ov = results["overall"]
    cm = results["confusion"]

    print(f"\n{'='*65}")
    print(title)
    print(f"{'='*65}")
    print(f"  Samples   : {ov['n']}")
    print(f"  Accuracy  : {ov['accuracy']:.1%}")
    print(f"  Precision : {ov['precision']:.1%}   (flagged-AI that are actually AI)")
    print(f"  Recall    : {ov['recall']:.1%}   (actual AI texts that were caught)")
    print(f"  F1        : {ov['f1']:.4f}")
    print(f"\n  TP={cm['TP']}  FP={cm['FP']}  TN={cm['TN']}  FN={cm['FN']}")

    sc = results["scores"]
    print(f"  Score dist: mean={sc['mean']}  min={sc['min']}  max={sc['max']}")

    def breakdown(title, data):
        print(f"\n{title}")
        print("-" * 65)
        for k, m in data.items():
            if "note" in m:
                print(f"  {k:<28} (single-class)")
            else:
                print(f"  {k:<28} acc={m['accuracy']:.1%}  f1={m['f1']:.3f}  n={m['n']}")

    breakdown("BY MODEL",        results["by_model"])
    breakdown("BY DOMAIN",       results["by_domain"])
    breakdown("BY ATTACK TYPE",  results["by_attack"])
    print("=" * 65 + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate live site JS logic against RAID")
    parser.add_argument("--n",          type=int,  default=200)
    parser.add_argument("--domain",     type=str,  default=None)
    parser.add_argument("--model",      type=str,  default=None)
    parser.add_argument("--adversarial",action="store_true")
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--compare",    action="store_true", help="Also run Python port for side-by-side comparison")
    parser.add_argument("--output",     type=str,  default=None)
    args = parser.parse_args()

    # Check Node.js is available
    # Find Node.js — check common Windows install path if not on PATH
    node_cmd = "node"
    try:
        subprocess.run([node_cmd, "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        fallback = r"C:\Program Files\nodejs\node.exe"
        if Path(fallback).exists():
            node_cmd = fallback
        else:
            print("ERROR: node is not installed or not in PATH.")
            sys.exit(1)

    samples = load_samples(args.n, args.domain, args.model, args.adversarial, args.seed)
    if not samples:
        print("No samples loaded — check filters.")
        sys.exit(1)

    ai_count    = sum(s["label"] for s in samples)
    human_count = len(samples) - ai_count
    print(f"Loaded {len(samples)} samples  (AI={ai_count}, Human={human_count})")
    print(f"Running through JS detector (node_runner.js -> index.html logic)...")

    node_results = run_node(samples, node_cmd=node_cmd)
    results      = evaluate(samples, node_results)
    print_results(results)

    if args.compare:
        print("Running Python port for comparison...")
        sys.path.insert(0, str(SCRIPT_DIR))
        from detector import score_text as py_score
        py_true, py_pred = [], []
        for s in tqdm(samples, desc="Python port"):
            r    = py_score(s["text"])
            py_pred.append(r["prediction"])
            py_true.append(s["label"])
        py_metrics = compute_metrics(py_true, py_pred)
        print(f"\n{'='*65}")
        print("PYTHON PORT RESULTS (for comparison)")
        print(f"{'='*65}")
        print(f"  Accuracy  : {py_metrics['accuracy']:.1%}")
        print(f"  Precision : {py_metrics['precision']:.1%}")
        print(f"  Recall    : {py_metrics['recall']:.1%}")
        print(f"  F1        : {py_metrics['f1']:.4f}")
        print(f"{'='*65}\n")

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
