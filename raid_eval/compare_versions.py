"""
Feature-level comparison between the live JS detector (index.html) and the
Python port (detector.py) on the same RAID samples.

Outputs:
  - Per-signal average scores for TP / FN / TN groups in both versions
  - Where the two versions agree vs. disagree
  - Which signals fire differently between JS and Python

Usage:
    python compare_versions.py --n 400
    python compare_versions.py --n 400 --domain news
"""

import argparse, json, os, random, subprocess, sys
from collections import defaultdict
from pathlib import Path

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    from datasets import load_dataset
except ImportError:
    print("pip install datasets scikit-learn tqdm"); sys.exit(1)

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
NODE_RUNNER = SCRIPT_DIR / "node_runner_verbose.js"   # extended runner (created below)

sys.path.insert(0, str(SCRIPT_DIR))
from detector import score_text as py_score

NODE_PATH = r"C:\Program Files\nodejs\node.exe" if Path(r"C:\Program Files\nodejs\node.exe").exists() else "node"


# ── Data ────────────────────────────────────────────────────────────────────

def load_samples(n, domain, model, adversarial, seed):
    random.seed(seed)
    print("Streaming RAID from HuggingFace...")
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    samples = []
    for row in ds:
        rm = row.get("model","unknown"); rd = row.get("domain","unknown"); ra = row.get("attack","none")
        if domain and rd != domain: continue
        if model  and rm != model:  continue
        if adversarial and ra in (None,"none",""): continue
        samples.append({"id": len(samples), "text": row.get("generation",""),
                        "label": 0 if rm=="human" else 1, "model": rm, "domain": rd, "attack": ra})
        if len(samples) >= n * 4: break
    random.shuffle(samples)
    return samples[:n]


# ── JS verbose runner ────────────────────────────────────────────────────────

def write_verbose_runner():
    """Write a version of node_runner that also returns per-signal scores."""
    src = Path(SCRIPT_DIR / "node_runner.js").read_text(encoding="utf-8")

    extra = """
// Verbose mode: also return individual signal scores
function scoreTextVerbose(text) {
  if (!text || text.trim().split(/\\s+/).length < 20)
    return { score:50, verdict:'Insufficient', l1:50, l2:0, l5:50, signals:{} };

  const perplexity     = calcPerplexity(text);
  const burstiness     = calcBurstiness(text);
  const lexical        = calcLexicalDiversity(text);
  const { score: aiPhrases } = calcAIPhrases(text);
  const hedging        = calcHedging(text);
  const passive        = calcPassiveVoice(text);
  const transitions    = calcTransitions(text);
  const clauseDepth    = calcClauseDepth(text);
  const punctuation    = calcPunctuationVariance(text);
  const paraUniformity = calcParagraphUniformity(text);
  const rareWords      = calcRareWords(text);
  const formality      = calcFormalityShift(text);
  const ngramRep       = calcNgramRepetition(text);
  const openerDiv      = calcSentenceOpenerDiversity(text);
  const punctFinger    = calcPunctuationFingerprint(text);
  const vocabCluster   = calcVocabClustering(text);
  const dmEnsemble     = calcDensityMelodyEnsemble(text);
  const densityMelody  = dmEnsemble.score;
  const monteCarlo     = runMonteCarloAnalysis(text);
  const forensic       = runForensicAnalysis(text);
  const layer5         = runAuthoralConsistency(text);

  const weights = [0.01,0.11,0.06,0.07,0.01,0.01,0.03,0.02,0.07,0.12,0.02,0.12,0.04,0.05,0.07,0.07,0.07,0.05];
  const scores  = [perplexity.score, burstiness, lexical, aiPhrases, hedging, passive,
    transitions, clauseDepth, punctuation, paraUniformity, rareWords, formality,
    ngramRep, openerDiv, punctFinger, vocabCluster, densityMelody, monteCarlo.mean];

  const evidence      = scores.reduce((s,v,i) => s + Math.max(0, v-50)*weights[i], 0);
  const baseComposite = Math.min(100, Math.round(evidence * 5));
  const anchorIdx     = [1,3,11,15];
  const anchorsHot    = anchorIdx.filter(i => scores[i] > 70).length;
  const convBonus     = [0,0,5,12,18][Math.min(anchorsHot,4)];
  const smokingGun    = scores[3]>=100?30: scores[3]>85?15:0;
  const composite     = Math.min(100, baseComposite + convBonus + smokingGun);
  const combined      = Math.round(Math.min(100, composite*0.75 + forensic.score*0.15 + layer5.score*0.10));

  let verdict = combined>=50?'AI': combined>=20?'Mixed':'Human';

  return {
    score: combined, verdict,
    l1: composite, l2: forensic.score, l5: layer5.score,
    conv_bonus: convBonus, smoking_gun: smokingGun,
    signals: {
      perplexity: perplexity.score, burstiness, lexical_diversity: lexical,
      ai_phrases: aiPhrases, hedging, passive_voice: passive,
      transitions, clause_depth: clauseDepth, punctuation_variance: punctuation,
      paragraph_uniformity: paraUniformity, rare_words: rareWords,
      formality_shift: formality, ngram_repetition: ngramRep,
      sentence_opener_diversity: openerDiv, punctuation_fingerprint: punctFinger,
      vocab_clustering: vocabCluster, density_melody: densityMelody,
      monte_carlo: monteCarlo.mean,
    }
  };
}
"""

    # Replace the old stdin handler with one that calls scoreTextVerbose
    new_handler = """
let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { raw += chunk; });
process.stdin.on('end', () => {
  const cleaned = raw.replace(/^\\uFEFF/, '');
  let items;
  try { items = JSON.parse(cleaned); } catch(e) {
    process.stderr.write('JSON parse error: ' + e.message + '\\n'); process.exit(1);
  }
  const results = items.map(item => {
    try { return { id: item.id, ...scoreTextVerbose(item.text) }; }
    catch(e) { process.stderr.write('Score error id=' + item.id + ': ' + e.message + '\\n');
      return { id: item.id, score:50, verdict:'Error', l1:0, l2:0, l5:0, signals:{} }; }
  });
  process.stdout.write(JSON.stringify(results));
});
"""
    # Strip the old stdin block from the copied source
    cut = src.find("// ─── Batch processing")
    base = src[:cut] if cut > 0 else src
    verbose_src = base + extra + new_handler
    NODE_RUNNER.write_text(verbose_src, encoding="utf-8")


def run_js_verbose(samples, batch_size=100):
    all_results = {}
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        payload = json.dumps([{"id":s["id"],"text":s["text"]} for s in batch])
        proc = subprocess.run([NODE_PATH, str(NODE_RUNNER)],
                              input=payload.encode("utf-8"), capture_output=True, timeout=120)
        stderr = proc.stderr.decode().strip()
        if stderr:
            for line in stderr.splitlines()[:2]:
                if "non-fatal" not in line: print(f"  [node] {line}")
        if proc.returncode != 0: continue
        for r in json.loads(proc.stdout.decode("utf-8")):
            all_results[r["id"]] = r
    return all_results


# ── Analysis helpers ─────────────────────────────────────────────────────────

SIGNAL_NAMES = [
    "perplexity","burstiness","lexical_diversity","ai_phrases","hedging",
    "passive_voice","transitions","clause_depth","punctuation_variance",
    "paragraph_uniformity","rare_words","formality_shift","ngram_repetition",
    "sentence_opener_diversity","punctuation_fingerprint","vocab_clustering",
    "density_melody","monte_carlo",
]

PY_SIGNAL_MAP = {          # Python port key → canonical name
    "perplexity_proxy":         "perplexity",
    "burstiness":               "burstiness",
    "lexical_diversity":        "lexical_diversity",
    "ai_phrases":               "ai_phrases",
    "hedging":                  "hedging",
    "passive_voice":            "passive_voice",
    "transitions":              "transitions",
    "punctuation_variance":     "punctuation_variance",
    "paragraph_uniformity":     "paragraph_uniformity",
    "rare_words":               "rare_words",
    "formality_shift":          "formality_shift",
    "ngram_repetition":         "ngram_repetition",
    "sentence_opener_diversity":"sentence_opener_diversity",
    "punctuation_fingerprint":  "punctuation_fingerprint",
    "vocab_clustering":         "vocab_clustering",
    "density_melody":           "density_melody",
}

def avg(lst): return round(sum(lst)/len(lst), 1) if lst else 0.0

def group_results(samples, js_results, py_results):
    """Sort samples into TP/FN/TN/FP buckets for both versions."""
    groups = {"js": defaultdict(list), "py": defaultdict(list)}

    for s in samples:
        jr = js_results.get(s["id"])
        pr = py_results.get(s["id"])
        if jr is None or pr is None: continue

        js_pred = 0 if jr["verdict"] == "Human" else 1
        py_pred = pr["prediction"]
        true    = s["label"]

        def bucket(pred): return ("TP" if true==1 and pred==1 else
                                  "FN" if true==1 and pred==0 else
                                  "TN" if true==0 and pred==0 else "FP")

        js_sig = jr.get("signals", {})
        py_sig = {PY_SIGNAL_MAP.get(k,k): v for k,v in pr.get("signals",{}).items()}

        entry = lambda sig, extra: {**sig, **extra, "model": s["model"]}

        groups["js"][bucket(js_pred)].append(entry(js_sig, {
            "combined": jr["score"], "l1": jr["l1"], "l2": jr["l2"], "l5": jr["l5"],
            "conv_bonus": jr.get("conv_bonus",0), "smoking_gun": jr.get("smoking_gun",0),
        }))
        groups["py"][bucket(py_pred)].append(entry(py_sig, {
            "combined": pr["combined_score"], "l1": pr["l1_score"],
            "l2": pr["l2_score"], "l5": pr["l5_score"],
            "conv_bonus": 0, "smoking_gun": 0,
        }))

    return groups


def signal_avgs(entries, keys):
    return {k: avg([e.get(k,0) for e in entries]) for k in keys}


def bar(v, w=25): return "[" + "#"*int(round(v/100*w)) + "."*(w-int(round(v/100*w))) + f"] {v:5.1f}"


# ── Report ───────────────────────────────────────────────────────────────────

def print_report(samples, js_results, py_results):
    groups = group_results(samples, js_results, py_results)

    def counts(ver):
        g = groups[ver]
        return len(g["TP"]), len(g["FP"]), len(g["TN"]), len(g["FN"])

    js_tp,js_fp,js_tn,js_fn = counts("js")
    py_tp,py_fp,py_tn,py_fn = counts("py")
    total = js_tp+js_fp+js_tn+js_fn

    def pct(num, denom): return f"{num/denom:.1%}" if denom else "n/a"

    print(f"\n{'='*70}")
    print("OVERALL COMPARISON  (same {total} samples run through both versions)")
    print(f"{'='*70}")
    print(f"  {'Metric':<22}  {'Live JS':>10}  {'Python Port':>12}")
    print(f"  {'-'*46}")
    ai_total = js_tp+js_fn
    print(f"  {'Accuracy':<22}  {pct(js_tp+js_tn,total):>10}  {pct(py_tp+py_tn,total):>12}")
    print(f"  {'Precision':<22}  {pct(js_tp,js_tp+js_fp):>10}  {pct(py_tp,py_tp+py_fp):>12}")
    print(f"  {'Recall':<22}  {pct(js_tp,js_tp+js_fn):>10}  {pct(py_tp,py_tp+py_fn):>12}")
    print(f"  {'True Positives':<22}  {js_tp:>10}  {py_tp:>12}")
    print(f"  {'False Positives':<22}  {js_fp:>10}  {py_fp:>12}")
    print(f"  {'True Negatives':<22}  {js_tn:>10}  {py_tn:>12}")
    print(f"  {'False Negatives':<22}  {js_fn:>10}  {py_fn:>12}")

    # Agreement
    agree = sum(
        1 for s in samples
        if js_results.get(s["id"]) and py_results.get(s["id"]) and
           (0 if js_results[s["id"]]["verdict"]=="Human" else 1) == py_results[s["id"]]["prediction"]
    )
    print(f"\n  Agreement between versions: {agree}/{total} ({agree/total:.1%})")

    # ── Architecture differences ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ARCHITECTURE DIFFERENCES")
    print(f"{'='*70}")
    rows = [
        ("Signals","18 (+ Monte Carlo)","16 (no Monte Carlo)"),
        ("Monte Carlo sampling","YES — 8-20 random subsets","NO — not implemented"),
        ("Clause depth","YES — subordinate clause scan","NO — not in Python port"),
        ("Convergence bonus","YES — +5/12/18 when anchors cluster","NO — flat weighted sum only"),
        ("Smoking-gun bonus","YES — +15/30 on saturated AI phrases","NO — phrases just add to score"),
        ("Evidence model","Only scores > 50 contribute (gap model)","Full weighted sum (all scores count)"),
        ("AI verdict threshold",">= 50 = AI  (aggressive)",">= 60 = AI  (conservative)"),
        ("Mixed threshold","20-49 = Mixed -> flagged as AI","40-59 = Mixed -> flagged as AI"),
        ("Human threshold","< 20 = Human","< 40 = Human"),
        ("Layer weights","L1 75% + L2 15% + L5 10%","L1 75% + L2 15% + L5 10% (same)"),
    ]
    for feat, js_val, py_val in rows:
        print(f"\n  {feat}")
        print(f"    JS  : {js_val}")
        print(f"    Py  : {py_val}")

    # ── Per-signal comparison: AI texts (TP+FN) ──────────────────────────────
    print(f"\n{'='*70}")
    print("SIGNAL SCORES ON AI TEXTS  (TP = caught, FN = missed)")
    print("Higher = more AI-like.  Gap = JS avg minus Python avg.")
    print(f"{'='*70}")

    js_tp_ents = groups["js"]["TP"]; js_fn_ents = groups["js"]["FN"]
    py_tp_ents = groups["py"]["TP"]; py_fn_ents = groups["py"]["FN"]

    sig_keys = SIGNAL_NAMES
    js_tp_avg = signal_avgs(js_tp_ents, sig_keys)
    js_fn_avg = signal_avgs(js_fn_ents, sig_keys)
    py_tp_avg = signal_avgs(py_tp_ents, sig_keys)
    py_fn_avg = signal_avgs(py_fn_ents, sig_keys)

    print(f"\n  {'Signal':<28} {'JS-TP':>6} {'JS-FN':>6} {'Py-TP':>6} {'Py-FN':>6}  Gap")
    print(f"  {'-'*62}")
    for sig in sig_keys:
        jtp = js_tp_avg.get(sig,0); jfn = js_fn_avg.get(sig,0)
        ptp = py_tp_avg.get(sig,0); pfn = py_fn_avg.get(sig,0)
        if sig == "monte_carlo":
            print(f"  {sig:<28} {jtp:6.1f} {jfn:6.1f} {'N/A':>6} {'N/A':>6}  (JS only)")
            continue
        if sig == "clause_depth":
            print(f"  {sig:<28} {jtp:6.1f} {jfn:6.1f} {'N/A':>6} {'N/A':>6}  (JS only)")
            continue
        gap = jtp - ptp
        flag = " *" if abs(gap) > 15 else ""
        print(f"  {sig:<28} {jtp:6.1f} {jfn:6.1f} {ptp:6.1f} {pfn:6.1f}  {gap:+.1f}{flag}")

    # ── JS-exclusive features deep dive ─────────────────────────────────────
    print(f"\n{'='*70}")
    print("JS-EXCLUSIVE FEATURES: HOW MUCH DO THEY MOVE THE SCORE?")
    print(f"{'='*70}")

    all_js = groups["js"]["TP"] + groups["js"]["FN"] + groups["js"]["TN"]
    bonus_avg = avg([e.get("conv_bonus",0) for e in all_js])
    gun_avg   = avg([e.get("smoking_gun",0) for e in all_js])
    tp_bonus  = avg([e.get("conv_bonus",0) for e in groups["js"]["TP"]])
    fn_bonus  = avg([e.get("conv_bonus",0) for e in groups["js"]["FN"]])
    tp_gun    = avg([e.get("smoking_gun",0) for e in groups["js"]["TP"]])
    fn_gun    = avg([e.get("smoking_gun",0) for e in groups["js"]["FN"]])

    print(f"\n  Convergence Bonus  (fires when 3-4 anchor signals all > 70)")
    print(f"    Overall avg : +{bonus_avg:.1f} pts")
    print(f"    TP (caught) : +{tp_bonus:.1f} pts")
    print(f"    FN (missed) : +{fn_bonus:.1f} pts   <- low bonus = anchor signals not clustering")

    print(f"\n  Smoking-Gun Bonus  (fires on saturated AI phrase score)")
    print(f"    Overall avg : +{gun_avg:.1f} pts")
    print(f"    TP (caught) : +{tp_gun:.1f} pts")
    print(f"    FN (missed) : +{fn_gun:.1f} pts")

    mc_tp = avg([e.get("monte_carlo",0) for e in groups["js"]["TP"]])
    mc_fn = avg([e.get("monte_carlo",0) for e in groups["js"]["FN"]])
    mc_tn = avg([e.get("monte_carlo",0) for e in groups["js"]["TN"]])
    print(f"\n  Monte Carlo Mean Score (random subset sampling)")
    print(f"    TP (caught AI) : {bar(mc_tp)}")
    print(f"    FN (missed AI) : {bar(mc_fn)}")
    print(f"    TN (human)     : {bar(mc_tn)}")

    # ── Verdict threshold impact ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICT THRESHOLD IMPACT  (where the two versions diverge)")
    print(f"{'='*70}")
    disagree_js_ai_py_human = []
    disagree_js_human_py_ai = []
    for s in samples:
        jr = js_results.get(s["id"]); pr = py_results.get(s["id"])
        if not jr or not pr: continue
        js_pred = 0 if jr["verdict"]=="Human" else 1
        py_pred = pr["prediction"]
        if js_pred == 1 and py_pred == 0:
            disagree_js_ai_py_human.append((s, jr, pr))
        elif js_pred == 0 and py_pred == 1:
            disagree_js_human_py_ai.append((s, jr, pr))

    print(f"\n  JS says AI, Python says Human: {len(disagree_js_ai_py_human)} samples")
    if disagree_js_ai_py_human:
        sub = disagree_js_ai_py_human[:5]
        true_pos = sum(1 for s,_,_ in disagree_js_ai_py_human if s["label"]==1)
        print(f"    Of these, {true_pos} are actually AI ({true_pos/len(disagree_js_ai_py_human):.0%})")
        print(f"    Sample scores (JS / Python):")
        for s,jr,pr in sub:
            print(f"      [{s['model']:12s}]  JS={jr['score']:3d}%  Py={pr['combined_score']:5.1f}  true={'AI' if s['label']==1 else 'Human'}")

    print(f"\n  JS says Human, Python says AI: {len(disagree_js_human_py_ai)} samples")
    if disagree_js_human_py_ai:
        true_pos = sum(1 for s,_,_ in disagree_js_human_py_ai if s["label"]==1)
        print(f"    Of these, {true_pos} are actually AI ({true_pos/len(disagree_js_human_py_ai):.0%})")
        print(f"    Sample scores (JS / Python):")
        for s,jr,pr in list(disagree_js_human_py_ai)[:5]:
            print(f"      [{s['model']:12s}]  JS={jr['score']:3d}%  Py={pr['combined_score']:5.1f}  true={'AI' if s['label']==1 else 'Human'}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY: KEY DIFFERENCES")
    print(f"{'='*70}")
    print("""
  Live JS is more aggressive (lower threshold, non-linear bonuses):
    + Higher recall — catches more AI text
    - Higher false-positive rate — flags more human text as AI
    + Monte Carlo adds a unique signal not in Python port
    + Convergence/smoking-gun bonuses give non-linear boosts on obvious AI
    - Gap-model evidence (only >50 contributes) means subtle AI signals are ignored

  Python Port is more conservative (higher threshold, linear scoring):
    + Higher precision — fewer false positives on human text
    - Lower recall — misses more AI text (especially MPT / Llama-chat)
    + Simpler to tune — flat weighted sum, no hidden non-linear effects
    - Missing Monte Carlo, clause depth, convergence/smoking-gun bonuses
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=400)
    parser.add_argument("--domain",     type=str, default=None)
    parser.add_argument("--model",      type=str, default=None)
    parser.add_argument("--adversarial",action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    write_verbose_runner()
    samples = load_samples(args.n, args.domain, args.model, args.adversarial, args.seed)
    if not samples: print("No samples."); sys.exit(1)
    print(f"Loaded {len(samples)} samples  (AI={sum(s['label'] for s in samples)}, Human={sum(1-s['label'] for s in samples)})")

    print("Scoring with JS (node_runner_verbose.js)...")
    js_results = {}
    for i in tqdm(range(0, len(samples), 100), desc="JS batches"):
        batch = samples[i:i+100]
        payload = json.dumps([{"id":s["id"],"text":s["text"]} for s in batch])
        proc = subprocess.run([NODE_PATH, str(NODE_RUNNER)],
                              input=payload.encode("utf-8"), capture_output=True, timeout=120)
        stderr = proc.stderr.decode().strip()
        if stderr:
            for line in stderr.splitlines()[:2]:
                if "non-fatal" not in line: print(f"  [node] {line}")
        if proc.returncode == 0:
            for r in json.loads(proc.stdout.decode("utf-8")): js_results[r["id"]] = r

    print("Scoring with Python port...")
    py_results = {}
    for s in tqdm(samples, desc="Python"):
        r = py_score(s["text"])
        if "signals" in r: py_results[s["id"]] = r

    print_report(samples, js_results, py_results)


if __name__ == "__main__":
    main()
