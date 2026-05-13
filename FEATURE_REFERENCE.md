# AI Origin Detector — Feature Reference

**Version:** Current (May 2026)  
**File:** `index.html` (~2,300 lines, single-file vanilla JS app)  
**Purpose:** Multi-layer forensic tool for detecting AI-generated text, 100% client-side

---

## Architecture Overview

The tool runs four active detection layers. Each layer produces a 0–100 AI confidence score. Layers are combined into one final verdict score.

```
Layer 1 — Linguistic Analysis      16 vectors   55% of final score (text reliability weighted)
Layer 2 — Forensic Character Scan  6 checks     20% of final score (always active)
Layer 3 — File Metadata Analysis   5+ signals   25% of final score (only when file is dropped)
Layer 5 — Authorial Consistency    5 micro-habits  10% of final score
```

> Layer 4 (RAIDAR-style rewrite analysis) is planned but not yet implemented.

**Final score formula:**
```
combinedScore = L1 * 0.75 + L2 * 0.15 + L5 * 0.10

If file was dropped:
  trueCombined = combinedScore * 0.85 + metaScore * 0.15
Else:
  trueCombined = combinedScore
```

**Verdict thresholds:**
| Score | Verdict |
|-------|---------|
| >= 68% | Likely AI-Generated |
| 35–67% | Ambiguous / Mixed Origin |
| < 35% | Likely Human-Written |

**Text reliability factor** (short texts nudged toward 50% uncertainty):
| Word count | Reliability multiplier |
|------------|----------------------|
| < 100 words | 0.60 |
| < 300 words | 0.82 |
| < 600 words | 0.92 |
| 600+ words | 1.00 |

---

## Layer 1 — Linguistic Analysis (16 Vectors)

All 16 scores are combined as a weighted average. Weights sum to 1.00.

---

### 1. Perplexity Proxy
**Weight:** 0.02 | **Function:** `calcPerplexity()`

**How it works:** Measures variance in word lengths across the text. Low variance (uniform word lengths) is treated as an AI signal — the theory being that AI tends toward moderate, predictable word length distributions.

**Implementation:** Computes the statistical variance of per-word character lengths. Score = `100 - (variance * 12)`, clamped to 0–100.

**Effectiveness:** Poor. Consistently returns near-0 on all AI texts tested — real AI writes with diverse word lengths. This signal was originally weighted 0.04 and was reduced to 0.02 after multiple false negatives. Weight may be reduced further or replaced.

**Known issue:** This proxy conflates "perplexity" (a language modeling concept requiring a trained model) with surface-level word length statistics. They are not the same thing.

---

### 2. Burstiness
**Weight:** 0.12 | **Function:** `calcBurstiness()`

**How it works:** Measures the coefficient of variation (CV = std/mean) of sentence lengths in words. Human writers "burst" — they alternate short punchy sentences with long complex ones. AI produces more uniform sentence length distributions.

**Implementation:** Splits text into sentences, computes lengths in words, calculates CV. Lower CV = higher AI score. Mapping: CV < 0.2 → ~80%, CV 0.4–0.5 → ~50–60%, CV > 0.8 → ~10%.

**Effectiveness:** Best-performing structural signal. Reliably distinguishes formal AI prose which has characteristically flat sentence rhythms. Less effective when AI deliberately varies sentence length.

**Note:** The highest-weighted signal (0.12). Performance validated across multiple test cases.

---

### 3. Lexical Diversity (TTR)
**Weight:** 0.08 | **Function:** `calcLexicalDiversity()`

**How it works:** Type-Token Ratio (TTR) — unique words divided by total words. AI tends to reuse key phrases, producing slightly lower TTR than human prose on similar topics.

**Implementation:** Tokenizes text, counts unique tokens. Score = `100 - (TTR * 130)`.

**Effectiveness:** Moderate. TTR is topic-dependent — technical writing has lower TTR than narrative writing regardless of authorship. Works better as a secondary signal than standalone.

---

### 4. AI Phrase Fingerprinting
**Weight:** 0.07 | **Function:** `calcAIPhrases()`

**How it works:** Scans for 42 phrases that are statistically overrepresented in AI output based on empirical observation. Includes patterns like "it is worth noting", "delve into", "multifaceted", "in the realm of", "leveraging", "tapestry", "nuanced", "paradigm", "pivotal".

**Implementation:** Case-insensitive substring match against a fixed phrase list. Score scales with phrase density per 100 words: `min(100, density * 25)`.

**Effectiveness:** Weak against modern AI. These phrase patterns were characteristic of early GPT-3/4 output. Newer models (GPT-4o, Claude 3.5+) have been trained away from these clichés and rarely produce them. Consistently near-0 on modern AI text. Still worth keeping for older-model detection but should not be heavily weighted.

**AI phrases list (42 total):**
`in conclusion`, `it is worth noting`, `it is important to note`, `furthermore`, `moreover`, `in summary`, `to summarize`, `delve into`, `it is crucial`, `in the realm of`, `as we explore`, `it becomes evident`, `navigating`, `multifaceted`, `nuanced`, `at its core`, `it is essential`, `foster`, `leveraging`, `paradigm`, `tapestry`, `underscore`, `pivotal`, `utilize`, `in today's world`, `in the context of`, `underpins`, `embodies`, `holistic`, `robust`, `streamline`, `synergy`, `it goes without saying`, `needless to say`, `when it comes to`, `let us`, `one can argue`, `it should be noted`, `in light of`, `it is clear that`, `first and foremost`, `last but not least`

---

### 5. Hedging Language Density
**Weight:** 0.02 | **Function:** `calcHedging()`

**How it works:** Counts hedge words and phrases per 100 words. AI tends to over-hedge with qualifiers.

**Hedge word list:** `perhaps`, `possibly`, `might`, `may`, `could`, `seem`, `appears`, `generally`, `typically`, `often`, `usually`, `tend to`, `suggest`, `indicate`

**Implementation:** Regex matches per hedge word, rate per 100 words. Score = `min(100, rate * 18)`.

**Effectiveness:** Weak standalone. Hedging is also common in formal human writing (academic, legal, scientific). Works better in combination with other signals. Weighted low at 0.02.

---

### 6. Passive Voice Detection
**Weight:** 0.02 | **Function:** `calcPassiveVoice()`

**How it works:** Counts sentences containing passive constructions (`is/are/was/were/be/been/being + past participle`). High passive voice rate is associated with AI in formal contexts.

**Implementation:** Regex pattern per sentence, ratio of passive sentences. Score = `min(100, ratio * 160)`.

**Effectiveness:** Weak. Passive voice is common in human academic writing and varies widely by domain. Weighted low at 0.02 for this reason.

---

### 7. Transition Word Uniformity
**Weight:** 0.06 | **Function:** `calcTransitions()`

**How it works:** Measures density of transition words (`however`, `therefore`, `consequently`, `furthermore`, etc.). AI tends to use transitions at an unnaturally even cadence.

**Transition word list (15):** `however`, `therefore`, `thus`, `consequently`, `nevertheless`, `additionally`, `furthermore`, `moreover`, `in contrast`, `on the other hand`, `as a result`, `for example`, `for instance`, `in addition`, `similarly`

**Implementation:** Count matches per 100 words. Score = `min(100, rate * 22)`.

**Effectiveness:** Moderate. Strong signal in formal AI text which is over-reliant on signposting transitions. Less effective when AI uses fewer explicit transitions.

---

### 8. Clause Depth
**Weight:** 0.04 | **Function:** `calcClauseDepth()`

**How it works:** Counts subordinate clause markers per sentence as a proxy for sentence syntactic complexity. AI tends toward moderate, uniform clause depth — not too simple, not too complex.

**Subordinating words checked:** `which`, `that`, `although`, `because`, `since`, `while`, `when`, `where`, `if`, `unless`, `whereas`, `whether`

**Implementation:** Average subordinate markers per sentence. Scores highest when avg is in the "AI zone" of 0.8–2.5 markers/sentence. Score = `max(0, (2.5 - |avg - 1.6|) * 50)` within that range, 30 otherwise.

**Effectiveness:** Moderate. The "sweet spot" detection logic is theoretically sound but the range is wide. Works best as a tie-breaker signal.

---

### 9. Punctuation Variance
**Weight:** 0.10 | **Function:** `calcPunctuationVariance()`

**How it works:** Measures the coefficient of variation in punctuation density across paragraphs. AI distributes punctuation marks (commas, semicolons, colons, dashes, parentheses) very evenly across paragraphs. Human writing has irregular punctuation density — some paragraphs are heavily punctuated, others minimal.

**Implementation:** Counts punctuation marks per word per paragraph. Computes variance across paragraphs. Low variance → high AI score. Score = `80 - (variance * 400)`.

**Effectiveness:** Good. One of the more reliable uniformity-based signals. Falls back to 50 (neutral) on single-paragraph texts, which is a known limitation.

---

### 10. Paragraph Length Uniformity
**Weight:** 0.10 | **Function:** `calcParagraphUniformity()`

**How it works:** Measures CV of paragraph lengths in words. AI tends to produce paragraphs of very similar length; human writing varies.

**Implementation:** Get paragraphs split by double-newline, compute CV of word counts. Low CV → high AI score. Score = `90 - (CV * 80)`.

**Effectiveness:** Good on multi-paragraph text. Returns 50 (neutral) for single-paragraph input — a significant limitation since many pasted texts lack paragraph breaks.

---

### 11. Rare Word Usage
**Weight:** 0.04 | **Function:** `calcRareWords()`

**How it works:** Filters to content words (length > 3, not in common word list of ~90 words) and measures their TTR. Low variety in content words = repetitive = AI signal.

**Implementation:** Content-word TTR. Score = `85 - (rareTTR * 90)`.

**Effectiveness:** Weak. Highly domain-dependent. Topic-specialized texts (medical, legal, technical) have naturally low content-word TTR regardless of authorship.

---

### 12. Register Stability (Formality Uniformity)
**Weight:** 0.10 | **Function:** `calcFormalityShift()`

**How it works:** Counts informal markers (contractions, casual words like `gonna`, `basically`, `literally`) and formal markers (`therefore`, `notwithstanding`, `henceforth`, etc.) per sentence. Measures variance in informality score across sentences. AI maintains a very consistent register throughout a document; humans naturally shift tone.

**Informal markers:** `i'm`, `it's`, `don't`, `can't`, `won't`, `isn't`, `you're`, `we're`, `gonna`, `wanna`, `kinda`, `yeah`, `ok`, `stuff`, `things`, `pretty`, `really`, `actually`, `basically`, `literally`, `honestly`, `like`

**Formal markers:** `therefore`, `subsequently`, `notwithstanding`, `whilst`, `regarding`, `pertaining`, `aforementioned`, `henceforth`, `hereby`, `therein`, `pursuant`, `whereby`

**Implementation:** Per-sentence informal marker count → variance across sentences. Low variance → high AI score. Extra +20 if formal text has zero informal markers across 5+ sentences.

**Effectiveness:** Strong. Consistently the highest-scoring individual vector across all AI text test cases (often 90–95%). Register stability is very difficult for AI to fake because it requires deliberately introducing natural-sounding inconsistency. Weight increased from 0.08 to 0.10 after testing confirmed reliability.

---

### 13. N-gram Repetition
**Weight:** 0.05 | **Function:** `calcNgramRepetition()`

**How it works:** Counts repeated 3-word sequences (trigrams) as a fraction of all unique trigrams. Humans rarely repeat exact 3-word phrases; AI recycles sentence templates.

**Implementation:** Build trigram frequency map, count trigrams appearing > 1 time, compute ratio. Score = `min(100, repetitionRate * 300)`.

**Effectiveness:** Moderate on longer texts, unreliable on short texts (< 100 words). High repetition in the synthetic Kaggle dataset made this appear effective there, but on real AI text (student essay style) repetition rates are similar to human writing.

---

### 14. Sentence Opener Diversity
**Weight:** 0.05 | **Function:** `calcSentenceOpenerDiversity()`

**How it works:** Extracts the first 2 words of each sentence and measures what fraction are unique. AI tends to reuse sentence-opening patterns.

**Implementation:** Set of unique 2-word openers / total sentences. Low diversity → high AI score. Score = `90 - (diversity * 95)`.

**Effectiveness:** Moderate. Less reliable because human writers in structured contexts (essays, reports) also tend toward repetitive openers. Works better on longer, less structured texts.

---

### 15. Punctuation Fingerprinting
**Weight:** 0.07 | **Function:** `calcPunctuationFingerprint()`

**How it works:** Identifies punctuation patterns characteristic of AI stylistic habits, specifically:
- Heavy em-dash (`—`) usage (AI's favorite interruption device)
- Consistent parenthetical rate
- Zero exclamation marks in longer formal text

**Implementation:** Counts per-sentence rates for em-dashes, parenthetical clauses, and exclamation marks. Additive scoring based on thresholds.

**Effectiveness:** Moderate. Very effective when text was generated by early Claude versions (which overused em-dashes). Less reliable on texts without em-dashes. The exclamation mark check is a weak signal.

---

### 16. Vocabulary Clustering
**Weight:** 0.06 | **Function:** `calcVocabClustering()`

**How it works:** Identifies "key terms" (words appearing 2+ times, length > 4 chars) and measures how evenly distributed they are across paragraphs. AI distributes domain vocabulary uniformly; humans cluster domain terms in relevant sections.

**Implementation:** Computes per-paragraph key-term density, then CV of those densities. Low CV → even distribution → AI. Score = `85 - (CV * 75)`.

**Effectiveness:** Good on multi-paragraph text. Falls back to sentence-split analysis on single-paragraph input. Theoretical basis is sound — humans introduce topics in bursts.

---

## Layer 2 — Forensic Character Scan

**Weight in final score:** 20%  
**Function:** `runForensicAnalysis()` + `calcForensicScore()`

This layer scans the raw Unicode character stream for artifacts that survive text editing and indicate machine-generated or machine-processed content. Unlike linguistic signals, forensic signals are binary — they're either there or they're not.

### Checks performed:

**1. Invisible Characters (23 types)**
Scans for zero-width spaces, joiners, directional marks, BOM characters, Hangul fillers, math invisible operators, and combining grapheme joiners. High-risk types score up to 15 points each (capped at 60 total). These characters are common in LLM copy-paste artifacts and some watermarking schemes.

High-risk examples: `U+200B` (Zero-Width Space, common in ChatGPT output), `U+FEFF` (BOM/Zero-Width No-Break Space), `U+2060` (Word Joiner, used in LLM watermarking), `U+034F` (Combining Grapheme Joiner, used in steganographic watermarks)

**2. Homoglyphs (32 types)**
Scans for Cyrillic, Greek, and other script characters that look identical to Latin letters (e.g., Cyrillic `а` vs Latin `a`). Used in both watermarking and evasion schemes. Each detected homoglyph type adds up to 8 points (capped at 40).

**3. Non-Standard Spaces**
Counts non-breaking spaces (`U+00A0`), narrow no-break spaces, and em-spaces. >3 non-breaking spaces is flagged as a common AI/auto-format artifact.

**4. Quote Style Mixing**
Detects mixed smart quotes (`"`) and straight quotes (`"`). Mixed styles indicate a copy-paste boundary where AI text was inserted into human-written text (or vice versa).

**5. Dash Type Mixing**
Detects em-dash (`—`), en-dash (`–`), and hyphen (`-`) mixing in the same document.

**6. Ellipsis Style Mixing**
Detects mixing of Unicode ellipsis (`…`) with straight dots (`...`).

**Scoring:**
| Finding | Score added |
|---------|------------|
| Each high-risk invisible char type | +15 (cap 60) |
| Each medium-risk invisible char | +5 (cap 20) |
| >3 non-breaking spaces | +15 |
| Homoglyph detected | +8 per type (cap 40) |
| Mixed quote styles | +12 |
| Mixed dash types | +8 |
| Mixed ellipsis styles | +6 |
| Unicode ellipsis only (no mix) | +4 |

**Effectiveness:** Highly reliable when anomalies are present — these are hard physical facts about the file's character encoding, not probabilistic estimates. The limitation is that most casual AI text (pasted from a browser) won't trigger these. Strongest use case: detecting professionally obfuscated or watermarked AI text, or copy-paste artifacts in submitted documents.

---

## Layer 3 — File Metadata Analysis

**Weight in final score:** 15% boost (only active when a DOCX or PDF is dropped)  
**Function:** `analyzeMetadata()` + `renderMetaPanel()`

Only runs when a file is dropped onto the file drop zone. Extracts metadata embedded in the file itself — data that reveals how the document was created and by whom.

### For DOCX files (parsed in-browser via ZIP/XML):

| Signal | What it reveals |
|--------|----------------|
| Author field | Name of the creating account |
| Total editing time | Very low time (< 2 min) on a long document suggests AI generation + quick submit |
| Revision session count (rsid) | Each unique editing session adds an rsid. Single rsid = one session (AI paste). Many rsids = document evolved over time (human) |
| Tracked changes | Insertions/deletions/moves from revision history |
| Created vs. modified timestamps | Gap of seconds to minutes on a long doc is suspicious |
| Application name | Word, Google Docs, LibreOffice, etc. |

### For PDF files (parsed via PDF.js CDN):

| Signal | What it reveals |
|--------|----------------|
| Author/Creator fields | Creating account or software |
| Producer field | PDF generator — some AI tools leave identifiable producers |
| CreationDate vs ModDate | Narrow gap = fast creation |
| Page count vs. creation time | Implied words-per-minute of "writing" |

**Effectiveness:** Very high when metadata is available and unmodified. The editing time signal is particularly strong — a 2,000-word essay with 45 seconds of editing time is definitively suspicious. Limitation: students who paste AI text into Word and then type a few sentences will accumulate legitimate-looking editing time. Also doesn't run on plain text paste.

---

## Layer 5 — Authorial Consistency Fingerprinting

**Weight in final score:** 10%  
**Function:** `runAuthoralConsistency()` + `renderLayer5Panel()`

Measures the consistency of micro-writing habits across paragraphs. The key insight: AI applies stylistic rules uniformly throughout a document (100% consistent Oxford comma usage, constant contraction rate, uniform paragraph openers). Human writers naturally drift — they're consistent enough to have a style but not perfectly so.

This layer scores *consistency of the habit*, not the habit itself. A writer who never uses contractions scores the same as one who always does — both are human patterns. Only the writer who uses contractions at a suspiciously uniform rate per paragraph is flagged.

### 5a. Contraction Rate Consistency (weight: 25%)
Measures the CV of contraction frequency across paragraphs. Needs 3+ paragraphs and an average contraction rate > 0.5% to activate (formal texts with no contractions fall back to neutral 30).

### 5b. Oxford Comma Consistency (weight: 20%)
Counts serial lists with and without the Oxford comma. Perfectly consistent usage in either direction is an AI signal; mixing both patterns is a human signal. Needs 2+ list instances to activate.

### 5c. Number Formatting Consistency (weight: 20%)
Checks whether numbers 1–12 appear as words ("three") or digits ("3") consistently. Perfect consistency across a long document is unusual for humans.

### 5d. Sentence-Final Preposition Consistency (weight: 15%)
Measures the rate of sentences ending in prepositions per paragraph. AI applies prescriptive grammar rules very uniformly; humans drift between formal and informal usage within a document.

### 5e. Paragraph Opener Word-Class Consistency (weight: 20%)
Classifies the first word of each paragraph as an article, pronoun, transition word, or other. High dominance of one class (> 45% of paragraphs) = AI signal. Requires 3+ paragraphs.

**Layer 5 composite:** Weighted average of 5 signals. Score >= 60 = suspicious, 35–59 = caution, < 35 = clean.

**Effectiveness:** Good on longer documents with 4+ paragraphs. Falls back to neutral (returns 50 on most signals) on short or single-paragraph texts. The contraction and opener signals have been most discriminating in testing. Layer 5 is the newest addition and has not been benchmarked separately yet.

---

## Scoring Model Summary

### Layer 1 weights (must sum to 1.00):

| # | Vector | Weight | Reliability |
|---|--------|--------|-------------|
| 1 | Perplexity proxy | 0.02 | Low — unreliable signal |
| 2 | Burstiness | 0.12 | High — best structural signal |
| 3 | Lexical diversity | 0.08 | Medium |
| 4 | AI phrase fingerprinting | 0.07 | Low-Medium — misses modern AI |
| 5 | Hedging density | 0.02 | Low |
| 6 | Passive voice | 0.02 | Low |
| 7 | Transition uniformity | 0.06 | Medium |
| 8 | Clause depth | 0.04 | Medium |
| 9 | Punctuation variance | 0.10 | High — reliable uniformity signal |
| 10 | Paragraph uniformity | 0.10 | High — reliable uniformity signal |
| 11 | Rare word usage | 0.04 | Low-Medium |
| 12 | Register stability | 0.10 | High — strongest individual signal |
| 13 | N-gram repetition | 0.05 | Medium |
| 14 | Sentence opener diversity | 0.05 | Medium |
| 15 | Punctuation fingerprinting | 0.07 | Medium |
| 16 | Vocabulary clustering | 0.06 | Medium |

---

## Benchmark Results

### Dataset: `andythetechnerd03/AI-human-text` (HuggingFace)
500 rows — student essay style, real AI vs human writing  
Fetched via HuggingFace datasets-server API (no login required)

| Metric | Result |
|--------|--------|
| Accuracy | 66.6% |
| Precision | 55.6% |
| Recall (AI detection rate) | 43.7% |
| F1 Score | 48.9% |
| Specificity (Human detection rate) | 79.8% |
| Baseline (predict all human) | 63.4% |

**Key finding:** The tool is better at confirming human authorship (79.8% specificity) than catching AI-generated text (43.7% recall). The primary failure mode is false negatives — AI text that the tool scores below 50.

**Root cause analysis:** This dataset contains AI-generated student essays designed to mimic informal human writing. The tool's strongest signals (register stability, AI phrases, paragraph uniformity) rely on detecting formal AI patterns. When AI is instructed to sound like a casual student, these signals return near-neutral.

**Score separation:** AI texts averaged 48.9, human texts averaged 47.8 — a 1-point gap, indicating the current signal set has limited discriminative power on informal AI text.

### Manual test cases (pre-benchmark):

| Text type | Score | Verdict | Correct? |
|-----------|-------|---------|----------|
| Basic AI essay (376 words) | 36% | Ambiguous | Partially — was previously 26% (wrong) |
| Human blog post (via Node test) | 31% | Human | Yes |
| AI informational text (289 words) | 36% | Ambiguous | Partially — borderline |

---

## Known Limitations

1. **Short text degrades all signals** — texts under 300 words are reliability-dampened toward 50% (uncertain). Most of the strongest signals (paragraph uniformity, vocabulary clustering, contraction consistency) require 3+ paragraphs to activate.

2. **Single-paragraph input** — many pasted texts lack double-newline paragraph breaks. Signals that depend on `getParagraphs()` fall back to 50 (neutral), significantly weakening the overall score.

3. **Informal AI text** — modern AI prompted to write in a casual, human-like style evades most linguistic signals. The tool performs best on formal AI writing (essays, reports, blog posts) and worse on AI-generated conversational or creative text.

4. **No trained model** — all signals are hand-crafted heuristics, not learned from labeled data. This makes the tool interpretable and fast but limits the ceiling on accuracy.

5. **Metadata requires file drop** — Layer 3 (often the most definitive signal) is completely absent for pasted text.

6. **AI phrases list aging** — the 42-phrase fingerprint list was compiled based on GPT-3/4 era output. Newer models produce these phrases much less frequently.

---

## Files

| File | Purpose |
|------|---------|
| `index.html` | Entire application (~2,300 lines) |
| `eval/run_eval.js` | Node.js evaluation pipeline |
| `eval/fetch_dataset.js` | HuggingFace API dataset downloader |
| `eval/real_dataset.csv` | 500-row benchmark dataset |
| `eval/results.csv` | Latest eval run output |
| `CLAUDE.md` | Development context for Claude Code |
| `AI_Detector_Roadmap.md` | Feature roadmap |
| `README.md` | GitHub project page |
