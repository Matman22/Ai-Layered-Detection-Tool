# Experiment Log

## Iteration 1 — +4 readability features

Candidates: fleschKincaid, gunningFog, functionWordRatio, commaRate (plain-JS, computed from raw text).
Config: LOSO logistic regression, LR=0.1, EPOCHS=3000, LAMBDA=0.01, standardized features, rank-sum AUROC. n=1369 rows.

| Held-out source    | (a) base 17 | (b) all 21 | (c) 4 candidates alone |
|--------------------|------------:|-----------:|-----------------------:|
| andythetechnerd03  | 0.6902 (n=500) | 0.6826 | 0.4039 |
| RAID               | 0.6968 (n=300) | 0.8008 | 0.4926 |
| MAGE               | 0.4507 (n=279) | 0.4625 | 0.3544 |
| HC3                | 0.2593 (n=290) | 0.2793 | 0.2684 |
| **MEAN**           | **0.5243**  | **0.5563** | **0.3798** |

Verdict: **KEEP** (0.5563 > 0.545 threshold; base-17 sanity check reproduced 0.5243 ≈ 0.524 baseline).

Interpretation: the readability features carry no transferable signal on their own (0.38 mean — anti-correlated on most sources), but interacting with the base 17 they lift RAID sharply (0.70→0.80) and nudge MAGE/HC3 up without hurting andythetechnerd03 much; note HC3 remains far below chance (0.28), so the inverted-polarity problem on that domain is untouched.

## Iteration 2 — quadratic (extremeness) terms

Hypothesis: feature direction is domain-specific but extremeness transfers; squared standardized terms (z², z fit on training sources only per fold, then re-standardized on train) let LR learn "extreme in either direction = AI". Script: loso_quad.js. Config otherwise identical (LR=0.1, EPOCHS=3000).

| Held-out source    | (a) 21 linear λ=0.01 | (b) 42 quad λ=0.01 | (b2) 42 quad λ=0.05 | (c) 21 squared alone λ=0.01 |
|--------------------|---------------------:|-------------------:|--------------------:|----------------------------:|
| andythetechnerd03  | 0.6826 (n=500)       | 0.5024             | 0.5088              | 0.4898                       |
| RAID               | 0.8008 (n=300)       | 0.8026             | 0.8142              | 0.6832                       |
| MAGE               | 0.4625 (n=279)       | 0.3514             | 0.3381              | 0.2146                       |
| HC3                | 0.2793 (n=290)       | 0.2634             | 0.2760              | 0.3136                       |
| **MEAN**           | **0.5563**           | **0.4799**         | **0.4843**          | **0.4253**                   |

Verdict: **REVERT** (best quad mean 0.4843 < 0.5563; sanity gate (a) reproduced 0.5563 exactly). Keep the 21-linear model from iteration 1.

Interpretation: the extremeness hypothesis fails — squared terms alone are *worse* than chance on 3 of 4 sources (mean 0.43), meaning what counts as "extreme" is itself domain-specific (each source occupies a different region of feature space, so held-out z² magnitudes don't transfer). The HC3 inversion did not improve (0.2793 → 0.2634/0.2760, essentially flat-to-worse); quad terms also destroyed andythetechnerd03 (0.68→0.50) and MAGE (0.46→0.35), with RAID the lone mild beneficiary (+0.01).

## Iteration 3 — direction-consistency feature selection

Hypothesis: features whose label direction (sign of within-source single-feature AUROC − 0.5) is inconsistent across the 3 training sources are the ones that invert on unseen domains. Per fold: compute per-training-source single-feature AUROCs; (A) strict = keep only if all 3 directions agree with |AUROC−0.5| ≥ 0.03; (B) majority = keep if ≥2 of 3 agree with margin ≥ 0.03. Script: loso_dircons.js. LR config identical (LR=0.1, EPOCHS=3000, LAMBDA=0.01).

| Held-out source    | (a) all-21 sanity | (A) strict | (B) majority |
|--------------------|------------------:|-----------:|-------------:|
| andythetechnerd03  | 0.6826 (n=500)    | 0.5000 (D=3) | 0.7143 (D=17) |
| RAID               | 0.8008 (n=300)    | N/A (D=0)  | 0.8266 (D=12) |
| MAGE               | 0.4625 (n=279)    | 0.3906 (D=3) | 0.5704 (D=13) |
| HC3                | 0.2793 (n=290)    | 0.3090 (D=1) | 0.3825 (D=16) |
| **MEAN**           | **0.5563**        | **N/A**    | **0.6235**   |

Selected features per fold:
- andythetechnerd03 held out — A: perplexity, burstiness, punctuation. B (17): perplexity, burstiness, lexical, aiPhrases, hedging, passive, transitions, punctuation, rareWords, ngramRep, openerDiv, punctFinger, vocabCluster, densityMelody, fleschKincaid, gunningFog, commaRate
- RAID held out — A: (none — no feature is direction-consistent across andythetechnerd03/MAGE/HC3). B (12): perplexity, burstiness, lexical, aiPhrases, passive, transitions, punctuation, ngramRep, densityMelody, fleschKincaid, gunningFog, functionWordRatio
- MAGE held out — A: aiPhrases, ngramRep, densityMelody. B (13): perplexity, burstiness, lexical, aiPhrases, passive, transitions, punctuation, rareWords, ngramRep, densityMelody, fleschKincaid, gunningFog, commaRate
- HC3 held out — A: passive. B (16): perplexity, burstiness, lexical, aiPhrases, hedging, passive, transitions, punctuation, ngramRep, openerDiv, punctFinger, vocabCluster, densityMelody, fleschKincaid, gunningFog, functionWordRatio

Verdict: **KEEP** — majority rule mean 0.6235 > 0.5563 (sanity gate reproduced 0.5563 exactly). New best: 21-feature pool with per-fold majority-direction selection (margin 0.03). Strict rule is unusable: with only 3 training sources it collapses to 0–3 features per fold (0 on the RAID fold — nothing agrees across andythetechnerd03/MAGE/HC3).

HC3 improved meaningfully (0.2793 → 0.3825, +0.10) though it remains inverted; every fold improved under majority (andy 0.68→0.71, RAID 0.80→0.83, MAGE 0.46→0.57 — now above chance). Eleven features survive majority selection in all 4 folds: perplexity, burstiness, lexical, aiPhrases, passive, transitions, punctuation, ngramRep, densityMelody, fleschKincaid, gunningFog; clauseDepth, formality, and paraUniformity are never selected. Anomaly worth noting: 9 features have single-feature AUROC of exactly 0.500 within andythetechnerd03 (perplexity, burstiness, punctuation, paraUniformity, formality, openerDiv, punctFinger, vocabCluster, commaRate) — verified: they are *constant* in that source (8 stuck at 50.0000, commaRate at 0.0000), i.e. the extractor never computed them for andythetechnerd03 rows. This silently disqualifies them from that source's "vote" and is a data-quality bug worth fixing upstream: perplexity and burstiness are top transferable features everywhere else.

## Iteration 4 — data fix: andythetechnerd03 frozen features

**Root cause (diagnosed, not a pipeline bug):** the upstream HF dataset `andythetechnerd03/AI-human-text` is itself pre-flattened — every row is lowercased with ALL punctuation and newlines stripped. Verified two ways:
1. Live datasets-server API returns flattened text at every offset probed (0, 50k, 150k, 300k, 450k of 462,873 rows; 0/10 rows at each offset contain a newline, comma, period, or uppercase letter). Sample at offset 0 (matches our row 1 byte-for-byte): `"studies have been proven that people are starting to not drive cars as much americans are buying fewer cars and getting fewer licenses there..."`
2. Our pipeline is faithful: `eval/real_dataset.csv` and `eval/combined_dataset.csv` contain the same flattened text (all 500 rows: 0 newlines, 0 commas, 0 periods, avg 2150 chars), while RAID/MAGE/HC3 rows in the same combined CSV retain normal punctuation. `fetch_dataset.js`/`fetch_datasets.js` never stripped anything.

Consequently the 9 frozen features are *unmeasurable* on this source: the index.html signal functions return their neutral 50 (commaRate computes a true 0) when text has no sentence/paragraph/punctuation structure. No re-fetch can fix this — perplexity and burstiness will never vary on andythetechnerd03. Bonus finding: paraUniformity is ALSO constant within MAGE and HC3 (only RAID varies), which is why it is never selected in any fold.

**Fix tested** (script: loso_dircons_v2.js; frozen (source,feature) pairs detected data-driven, not hardcoded): (C) explicitly exclude frozen pairs from that source's direction votes; (D) additionally treat frozen values as missing when that source is in TRAINING (impute mean of the other training sources → z≈0, rows stop distorting the weight). LR config unchanged (LR=0.1, EPOCHS=3000, LAMBDA=0.01).

| Held-out source | (a) all-21 | (B) majority (iter 3) | (C) +vote-excl | (D) +train-impute |
|---|---:|---:|---:|---:|
| andythetechnerd03 | 0.6826 | 0.7143 | 0.7143 | 0.7143 |
| RAID | 0.8008 | 0.8266 | 0.8266 | 0.8281 |
| MAGE | 0.4625 | 0.5704 | 0.5704 | 0.5637 |
| HC3 | 0.2793 | 0.3825 | 0.3825 | 0.3770 |
| **MEAN** | **0.5563** | **0.6235** | **0.6235** | **0.6208** |

Verdict: **NO CHANGE — keep iteration-3 config (0.6235)**. Sanity gates reproduced exactly (0.5563 / 0.6235). (C) is bit-identical to (B) in every fold and selects identical feature sets — a constant feature scores AUROC exactly 0.500, fails the 0.03 margin, and therefore already abstains from that source's vote; the majority rule was implicitly robust to the frozen features all along. (D) fails the gate (0.6208 < 0.6235): neutral-50 constants in training are apparently a *mildly useful* anchor rather than a distortion — imputing them away nudged RAID up (+0.0015) but cost MAGE (−0.0067) and HC3 (−0.0055).

Do perplexity/burstiness now vary on andythetechnerd03? **No, and they cannot** — the source text genuinely contains no punctuation/newlines, so any fix must come from replacing the source (e.g. swap in a non-flattened essay corpus) rather than re-fetching or re-extracting. That is a candidate for a future iteration.

## Iteration 5 — 5th training source (dmitva/human_ai_generated_text)

Hypothesis: adding a clean (punctuation-intact) 5th AI/human source to the TRAINING POOLS ONLY makes direction votes more reliable. Held-out test sets unchanged (dmitva is never held out), so the gate stays comparable.

**Source chosen:** `dmitva/human_ai_generated_text` (config `default`, split `train`, 1M rows) — first candidate probed, worked immediately. Each record has `human_text` + `ai_text` columns; exploded into two rows per record (HC3-style). Label mapping: `human_text` → 0, `ai_text` → 1. Evidence: sample text is punctuation/case/newline-intact; human samples contain organic typos ("desicions", "ncreased to lear", "Some Schools offter"), AI samples are polished template prose ("Ultimately, this decision will depend on the individual student..."). Fetched 250 human + 250 AI (offsets 0–300, min 20 words); `source5.csv`, features appended in `features_ext5.csv` (1369+500 rows). Scripts: fetch_source5.js, extract_ext5.js, loso_5src.js.

Configs: (s21)/(sB) 3-source sanity gates; (a5) all-21 linear with 4-source pool; (M3) majority = ≥3 of 4 sources agree with |AUROC−0.5| ≥ 0.03; (M2) ≥2 of 4 agree, 2-2 ties broken by pooled-train direction (kept only if pooled |AUROC−0.5| ≥ 0.03). LR config unchanged (LR=0.1, EPOCHS=3000, LAMBDA=0.01).

| Held-out source | (s21) 3-src all-21 | (sB) 3-src majority | (a5) 4-src all-21 | (M3) 4-src maj≥3 | (M2) 2-of-4+ties |
|---|---:|---:|---:|---:|---:|
| andythetechnerd03 | 0.6826 | 0.7143 | 0.6323 | 0.5943 (D=7) | 0.6491 (D=20) |
| RAID | 0.8008 | 0.8266 | 0.6406 | 0.6245 (D=6) | 0.7363 (D=16) |
| MAGE | 0.4625 | 0.5704 | 0.8628 | 0.9010 (D=10) | 0.8625 (D=17) |
| HC3 | 0.2793 | 0.3825 | 0.3565 | 0.3594 (D=7) | 0.4249 (D=17) |
| **MEAN** | **0.5563** | **0.6235** | **0.6231** | **0.6198** | **0.6682** |

Verdict: **KEEP — new best 0.6682** via (M2), the 2-of-4-with-tie-break variant (0.6682 > 0.6235 gate). Both sanity gates reproduced exactly (0.5563 / 0.6235). The originally hypothesized rule (M3, strict >half of 4 = ≥3 votes) FAILS the gate at 0.6198 — with dmitva's strong-but-often-inverted directions it shrinks selection to 6–10 features and drops andy/RAID hard.

HC3 movement: 0.3825 → 0.4249 (+0.042 under M2; still inverted but best HC3 ever recorded). MAGE is the big winner: 0.5704 → 0.9010 (M3) / 0.8625 (M2) — dmitva (student-essay domain, ChatGPT-style AI) is apparently distributionally close to MAGE and anchors its fold. Costs: RAID 0.8266 → 0.7363, andy 0.7143 → 0.6491 under M2 (the per-fold hypothesis "improves all 4 folds" is REFUTED — 2 folds up, 2 down; the mean gate passes on MAGE's +0.29).

Anomalies: (1) dmitva single-feature directions are strong but frequently inverted vs the other sources — lexical 0.022, rareWords 0.034, fleschKincaid 0.165, ngramRep 0.153 vs formality 0.933, burstiness 0.910 — so it adds discriminative rows to the pool but *disagrees* in direction votes, which is why loosening to 2-of-4 (D=16–20, nearly all-21) beats tightening to 3-of-4. The win comes mostly from the enlarged training pool, not sharper voting: (a5) all-21 with the pool already jumps MAGE 0.46 → 0.86. (2) M2 selects near-everything, so iteration 3's selection machinery contributes little on top of the pool at 4 sources. (3) Fragility caveat: the mean now leans on one fold (MAGE 0.90); a different 5th source could swing it.
