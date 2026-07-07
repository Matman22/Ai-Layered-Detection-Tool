"""
Gatekeeper eval for Fast-DetectGPT.

Scores every row of eval/combined_dataset.csv with fast_detect_score() and
reports AUROC. Because Fast-DetectGPT is zero-shot (no training), there is no
"leave one source out" fitting step — we just report AUROC PER SOURCE plus the
pooled number. That is the apples-to-apples comparison against the stylometric
classifier's cross-dataset result:

    stylometry  mean LOSO AUROC = 0.524  (chance; HC3 0.26 inverted)

If Fast-DetectGPT can't clearly beat ~0.52 per source, deploying a backend is
not worth it — do NOT wire it into index.html. If it does (paper reports ~0.99
in-domain; expect lower on this messy mix but well above chance), proceed.

Usage:
    pip install -r backend/requirements.txt
    python backend/eval_fastdetect.py [--limit N]   # --limit for a quick sample
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fast_detect import fast_detect_score   # noqa: E402

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "combined_dataset.csv")


def auroc(pairs):
    """AUROC via Mann-Whitney U. pairs = list of (score, label 0/1)."""
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    ordered = sorted(pairs, key=lambda p: p[0])
    # average ranks for ties
    ranks = [0.0] * len(ordered)
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sum_pos = sum(rank for rank, (_, y) in zip(ranks, ordered) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cap rows per source (0 = all)")
    args = ap.parse_args()

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_source = {}
    per_source_count = {}
    scored_all = []
    for idx, row in enumerate(rows):
        src = row.get("source", "all")
        if args.limit and per_source_count.get(src, 0) >= args.limit:
            continue
        text = (row.get("text_content") or "").strip()
        try:
            label = int(row["is_ai_generated"])
        except (KeyError, ValueError):
            continue
        r = fast_detect_score(text)
        if not r["ok"]:
            continue
        d = r["discrepancy"]
        by_source.setdefault(src, []).append((d, label))
        scored_all.append((d, label))
        per_source_count[src] = per_source_count.get(src, 0) + 1
        if (idx + 1) % 50 == 0:
            print(f"  scored {idx + 1}/{len(rows)}", end="\r", file=sys.stderr)

    print("\nFAST-DETECTGPT AUROC (zero-shot, gpt2)\n")
    aucs = []
    for src in sorted(by_source):
        a = auroc(by_source[src])
        aucs.append(a)
        n = len(by_source[src])
        print(f"  {src:20s} AUROC {a:.4f}  (n={n})")
    pooled = auroc(scored_all)
    mean_auc = sum(aucs) / len(aucs) if aucs else float("nan")
    print(f"\n  MEAN per-source AUROC {mean_auc:.4f}")
    print(f"  POOLED AUROC          {pooled:.4f}")
    print("\n  Compare: stylometry classifier mean LOSO AUROC = 0.524 (chance)")
    print("  Ship the backend only if Fast-DetectGPT is clearly higher.")


if __name__ == "__main__":
    main()
