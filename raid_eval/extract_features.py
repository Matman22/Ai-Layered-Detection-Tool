"""
Step 4 — feature extraction.

Runs every calc_* signal in detector.py over each row of the unified
multi-dataset CSV and writes a feature matrix for the stylometric classifier.

    python extract_features.py                          # eval/multi_dataset.csv -> eval/features.csv
    python extract_features.py --input ../eval/adversarial_raid.csv --output ../eval/features_adv.csv

Output columns: <18 signals>, is_ai_generated, source_dataset, model
"""

import argparse
import csv
from pathlib import Path

import detector as det
from tqdm import tqdm

csv.field_size_limit(10_000_000)

EVAL_DIR = Path(__file__).parent.parent / "eval"

# Same 18 signals the detector scores on. Order is the contract with train_stylometric.py.
SIGNALS = [
    ("perplexity_proxy", det.calc_perplexity_proxy),
    ("burstiness", det.calc_burstiness),
    ("lexical_diversity", det.calc_lexical_diversity),
    ("ai_phrases", det.calc_ai_phrases),
    ("hedging", det.calc_hedging),
    ("passive_voice", det.calc_passive_voice),
    ("transitions", det.calc_transitions),
    ("punctuation_variance", det.calc_punctuation_variance),
    ("paragraph_uniformity", det.calc_paragraph_uniformity),
    ("rare_words", det.calc_rare_words),
    ("formality_shift", det.calc_formality_shift),
    ("ngram_repetition", det.calc_ngram_repetition),
    ("sentence_opener_diversity", det.calc_sentence_opener_diversity),
    ("punctuation_fingerprint", det.calc_punctuation_fingerprint),
    ("vocab_clustering", det.calc_vocab_clustering),
    ("density_melody", det.calc_density_melody),
    ("forensic", det.calc_forensic_score),
    ("authorial", det.calc_authorial_score),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(EVAL_DIR / "multi_dataset.csv"))
    ap.add_argument("--output", default=str(EVAL_DIR / "features.csv"))
    ap.add_argument("--min-words", type=int, default=20)
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Missing {inp}. Run the fetcher first: node eval/fetch_dataset.js --multi --per-source 400 --balance")
        return

    feat_names = [n for n, _ in SIGNALS]
    header = feat_names + ["is_ai_generated", "source_dataset", "model"]

    with open(inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in tqdm(rows, desc="extracting features"):
        text = (r.get("text_content") or "").strip()
        if len(text.split()) < args.min_words:
            continue
        vec = [round(fn(text), 4) for _, fn in SIGNALS]
        out_rows.append(vec + [r["is_ai_generated"], r.get("source_dataset", ""), r.get("model", "")])

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} feature rows x {len(feat_names)} signals -> {args.output}")


if __name__ == "__main__":
    main()
