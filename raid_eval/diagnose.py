"""
Signal-level diagnostics: which signals fail on AI text that slips through?

Compares signal scores for:
  - True Positives  (AI correctly caught)
  - False Negatives (AI that slipped through as "Human")
  - True Negatives  (Human correctly passed)

Usage:
    python diagnose.py                    # 500 samples, all domains
    python diagnose.py --n 1000
    python diagnose.py --domain news
    python diagnose.py --adversarial
"""

import argparse
import json
import random
import sys
from collections import defaultdict

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets")
    sys.exit(1)

from sklearn.metrics import f1_score
from tqdm import tqdm

from detector import score_text

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SIGNAL_NAMES = [
    "perplexity_proxy",
    "burstiness",
    "lexical_diversity",
    "ai_phrases",
    "hedging",
    "passive_voice",
    "transitions",
    "punctuation_variance",
    "paragraph_uniformity",
    "rare_words",
    "formality_shift",
    "ngram_repetition",
    "sentence_opener_diversity",
    "punctuation_fingerprint",
    "vocab_clustering",
    "density_melody",
]


def load_samples(n, domain, model, adversarial, seed):
    random.seed(seed)
    print("Streaming RAID from HuggingFace...")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    samples = []
    for row in ds:
        rm = row.get("model", "unknown")
        rd = row.get("domain", "unknown")
        ra = row.get("attack", "none")
        if domain and rd != domain:
            continue
        if model and rm != model:
            continue
        if adversarial and ra in (None, "none", ""):
            continue
        samples.append({
            "text": row.get("generation", ""),
            "label": 0 if rm == "human" else 1,
            "model": rm,
            "domain": rd,
            "attack": ra,
        })
        if len(samples) >= n * 4:
            break
    random.shuffle(samples)
    return samples[:n]


def run(samples):
    groups = {"TP": [], "FN": [], "TN": [], "FP": []}

    for s in tqdm(samples, desc="Scoring"):
        r = score_text(s["text"])
        if "signals" not in r:
            continue
        sig = r["signals"]
        entry = {
            "combined": r["combined_score"],
            "signals": sig,
            "model": s["model"],
            "domain": s["domain"],
            "attack": s["attack"],
        }
        true, pred = s["label"], r["prediction"]
        if true == 1 and pred == 1:
            groups["TP"].append(entry)
        elif true == 1 and pred == 0:
            groups["FN"].append(entry)
        elif true == 0 and pred == 0:
            groups["TN"].append(entry)
        else:
            groups["FP"].append(entry)

    return groups


def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def signal_averages(entries):
    if not entries:
        return {}
    out = {}
    for sig in SIGNAL_NAMES:
        vals = [e["signals"].get(sig, 0.0) for e in entries]
        out[sig] = round(avg(vals), 1)
    out["combined"] = round(avg([e["combined"] for e in entries]), 1)
    return out


def bar(value, width=30, max_val=100):
    filled = int(round(value / max_val * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:5.1f}"


def print_report(groups):
    tp, fn, tn, fp = groups["TP"], groups["FN"], groups["TN"], groups["FP"]
    total = len(tp) + len(fn) + len(tn) + len(fp)
    ai_total = len(tp) + len(fn)
    human_total = len(tn) + len(fp)

    print("\n" + "=" * 70)
    print("SAMPLE BREAKDOWN")
    print("=" * 70)
    print(f"  Total samples : {total}")
    print(f"  AI texts      : {ai_total}  (TP={len(tp)}, FN={len(fn)})")
    print(f"  Human texts   : {human_total}  (TN={len(tn)}, FP={len(fp)})")
    recall = len(tp) / ai_total if ai_total else 0
    print(f"  Recall        : {recall:.1%}  ({len(fn)} AI texts slipped through)")

    tp_avg = signal_averages(tp)
    fn_avg = signal_averages(fn)
    tn_avg = signal_averages(tn)

    # Rank signals by gap between FN and TN (signals where FN looks human-like)
    gaps = {}
    for sig in SIGNAL_NAMES:
        fn_score = fn_avg.get(sig, 0)
        tn_score = tn_avg.get(sig, 0)
        tp_score = tp_avg.get(sig, 0)
        # How much lower is FN than TP? That's the "failure gap"
        gaps[sig] = tp_score - fn_score

    ranked = sorted(gaps.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("SIGNAL SCORES: TP (caught AI) vs FN (missed AI) vs TN (human)")
    print("Higher score = more AI-like. Gap = how much lower FN is vs TP.")
    print("=" * 70)
    print(f"  {'Signal':<30}  {'TP':>6}  {'FN':>6}  {'TN':>6}  {'Gap':>6}")
    print("-" * 70)
    for sig, gap in ranked:
        tp_s = tp_avg.get(sig, 0)
        fn_s = fn_avg.get(sig, 0)
        tn_s = tn_avg.get(sig, 0)
        flag = " <<" if gap > 10 else ""
        print(f"  {sig:<30}  {tp_s:6.1f}  {fn_s:6.1f}  {tn_s:6.1f}  {gap:+6.1f}{flag}")
    print("-" * 70)
    print(f"  {'COMBINED':<30}  {tp_avg.get('combined',0):6.1f}  {fn_avg.get('combined',0):6.1f}  {tn_avg.get('combined',0):6.1f}")

    print("\n" + "=" * 70)
    print("TOP FAILING SIGNALS (biggest gap between caught-AI and missed-AI)")
    print("=" * 70)
    for i, (sig, gap) in enumerate(ranked[:5], 1):
        fn_s = fn_avg.get(sig, 0)
        tp_s = tp_avg.get(sig, 0)
        tn_s = tn_avg.get(sig, 0)
        print(f"\n  #{i}  {sig}  (gap = {gap:+.1f})")
        print(f"       TP (caught AI)  {bar(tp_s)}")
        print(f"       FN (missed AI)  {bar(fn_s)}")
        print(f"       TN (human)      {bar(tn_s)}")

    # Model breakdown for FN
    if fn:
        print("\n" + "=" * 70)
        print("FALSE NEGATIVES BY MODEL (which LLMs evade detection most?)")
        print("=" * 70)
        fn_models = defaultdict(int)
        ai_models = defaultdict(int)
        for e in fn:
            fn_models[e["model"]] += 1
        for e in tp + fn:
            ai_models[e["model"]] += 1
        for model, total_ai in sorted(ai_models.items(), key=lambda x: -x[1]):
            fn_count = fn_models.get(model, 0)
            miss_rate = fn_count / total_ai if total_ai else 0
            print(f"  {model:<25}  missed {fn_count:3d}/{total_ai:3d}  ({miss_rate:.0%})")

    # Score distribution of FN
    if fn:
        fn_scores = sorted(e["combined"] for e in fn)
        print("\n" + "=" * 70)
        print("FALSE NEGATIVE SCORE DISTRIBUTION")
        print("(These are AI texts scored below 60 — should have been caught)")
        print("=" * 70)
        buckets = [0] * 6
        for s in fn_scores:
            idx = min(5, int(s // 10))
            buckets[idx] += 1
        for i, count in enumerate(buckets):
            label = f"{i*10}-{i*10+9}"
            bar_str = "#" * count
            print(f"  {label:8s}  {bar_str:<40} {count}")
        print(f"\n  Median FN score: {fn_scores[len(fn_scores)//2]:.1f}")
        print(f"  Mean FN score:   {avg(fn_scores):.1f}")

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    top_failing = [sig for sig, gap in ranked[:3] if gap > 5]
    for sig in top_failing:
        gap = gaps[sig]
        fn_s = fn_avg.get(sig, 0)
        tn_s = tn_avg.get(sig, 0)
        if fn_s < tn_s + 5:
            print(f"  - {sig}: FN ({fn_s:.1f}) is close to human baseline ({tn_s:.1f}).")
            print(f"    The signal fires correctly on easy AI but not on harder samples.")
            print(f"    Consider: tighten the scoring formula or increase its weight.")
        else:
            print(f"  - {sig}: Large gap ({gap:+.1f}) between caught vs missed AI.")
            print(f"    Missed AI scores {fn_s:.1f} vs caught AI {tp_avg.get(sig,0):.1f}.")
            print(f"    Consider: expand pattern coverage or lower the detection threshold.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--adversarial", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=str, default=None, help="Save raw group data to JSON")
    args = parser.parse_args()

    samples = load_samples(args.n, args.domain, args.model, args.adversarial, args.seed)
    if not samples:
        print("No samples loaded.")
        sys.exit(1)
    print(f"Loaded {len(samples)} samples")

    groups = run(samples)
    print_report(groups)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({k: v[:50] for k, v in groups.items()}, f, indent=2)
        print(f"Saved sample data to {args.save}")


if __name__ == "__main__":
    main()
