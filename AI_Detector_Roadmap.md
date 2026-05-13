# AI Origin Detector — Claude Code Context

## What This Project Is
A multi-layer forensic tool for detecting AI-generated text. Built entirely in
vanilla JavaScript — no frameworks, no server, no data sent anywhere. Runs 100%
client-side in the browser. Single file: `index.html`.

## Current Architecture

### Layer 1 — Linguistic Analysis (17 vectors)
Perplexity, burstiness, lexical diversity, AI phrase fingerprinting (42 phrases),
hedging density, passive voice, transition uniformity, clause depth, punctuation
variance, paragraph uniformity, rare word profiling, register stability, n-gram
repetition, sentence opener diversity, punctuation fingerprinting, vocabulary
clustering, and authorial consistency fingerprinting.

### Layer 2 — Forensic Character Scan
23 invisible Unicode character types, 32 Cyrillic/Greek homoglyph substitutions,
smart/straight quote mixing, dash type analysis, non-standard whitespace,
foreign script block detection.

### Layer 3 — File Metadata Analysis
Reads DOCX and PDF metadata in-browser via PDF.js (CDN). Extracts author fields,
total editing time, revision session markers (rsid), created/modified timestamps,
tracked changes (insertions/deletions), application/producer fingerprinting.

### Layer 5 — Authorial Consistency Fingerprinting
Measures micro-habit consistency across 5 signals: contraction usage, Oxford comma,
number formatting, sentence-final prepositions, paragraph opening word class.
Scores consistency of the habit, not the habit itself — so formal and casual
writers both score human for different reasons.

## Scoring Model
Unified weighted evidence model — not chained multipliers:
- Layer 1 (linguistic): 55% weight × text reliability factor
- Layer 2 (forensic): 20% weight, always active
- Layer 3 (metadata): 25% weight, only active when file is dropped
- Confidence penalty: short texts nudged toward 50% (uncertain)
- Text reliability: <100 words=0.6, <300=0.82, <600=0.92, 600+=1.0

## File Structure
```
index.html          ← entire app (2100+ lines, single file)
CLAUDE.md           ← this file
AI_Detector_Roadmap.md  ← feature roadmap and industry research
README.md           ← GitHub project documentation
```

## Tech Stack
- Vanilla JavaScript, HTML, CSS — zero build step, zero dependencies
- PDF.js loaded from CDN (only for PDF parsing)
- No npm, no bundler, no framework
- Open directly in browser — no server needed

## Deployment
- GitHub Pages: auto-deploys from main branch on push
- Live URL: https://yourusername.github.io/your-repo-name
- Single file — edit index.html, push, done

## How to Test Changes
```bash
# Open locally
open index.html          # Mac
start index.html         # Windows

# Push to GitHub Pages
git add .
git commit -m "description"
git push
```

## Known Issues / Watch Out For
- File is 2100+ lines — be surgical with edits, don't rewrite whole sections
- str_replace edits have dropped function declaration lines before (e.g. getLevel,
  AI_PHRASES const) — always verify surrounding context when editing
- All const declarations in runAnalysis() must be unique — browser throws
  "already declared" if duplicated
- PDF.js loads from CDN on first PDF drop — requires internet connection
- Scoring weights must sum to 1.0 across all 17 linguistic vectors

## Pending Features (Priority Order)
See AI_Detector_Roadmap.md for full details. Top priority:

1. RAIDAR-style rewriting analysis (Layer 4)
   - Send text to Anthropic API, ask it to rewrite, measure edit distance
   - Small changes = AI-generated. Large changes = human-written.
   - API infrastructure already available in artifact system

2. Semantic coherence mapping
   - Paragraph-to-paragraph lexical overlap
   - Callback density (do ideas circle back?)
   - Conclusion vocabulary novelty

3. Sentence-level heatmap visualization
   - Color-highlight individual sentences by AI confidence
   - Red = high AI, green = human
   - Makes the tool explainable, not a black box

4. Citation verification
   - Extract citations from text
   - Web search to verify they exist
   - Fabricated citations are a strong AI signal

5. Readability consistency scoring (Flesch-Kincaid per paragraph)

## Project Background
Built iteratively in a Claude.ai conversation. Key milestones:
- Started as 12 linguistic vectors
- Pen-tested with marine biology evasion text → found burstiness and register
  were too gameable → added n-gram repetition, opener diversity, punctuation
  fingerprinting, vocab clustering
- Added forensic Layer 2 for Unicode/hidden character detection
- Added file drop + Layer 3 metadata (DOCX XML parsing + PDF.js)
- Rebuilt scoring as unified weighted evidence model (was broken chained multipliers)
- Removed fake loading animation (was 3.68 seconds of artificial delay)
- Added Layer 5 authorial consistency fingerprinting

## Student Context
This is a portfolio project by a Drexel University MIS student. The goal is to
demonstrate NLP, digital forensics, and adversarial ML skills for co-op and
job applications. Keep code well-commented. Prefer clear readable logic over
clever one-liners. Document what each detection check does and why.

## Session Startup Checklist
When starting a new Claude Code session:
1. Read index.html to understand current state
2. Check AI_Detector_Roadmap.md for pending features
3. Ask what the user wants to work on
4. Make targeted edits — never rewrite large sections unnecessarily
5. Test by opening index.html in browser after changes
6. Commit with descriptive message when feature is complete
