# AI Origin Detector

A multi-layer forensic tool for detecting AI-generated text, 
built entirely in vanilla JavaScript — no server, no API, 
nothing sent anywhere.

https://matman22.github.io/Ai-Layered-Detection-Tool/ 

---

## How it works

### Layer 1 — Linguistic Analysis (18 inputs)
Perplexity modeling (3-proxy ensemble), sentence burstiness, 
lexical diversity, AI phrase fingerprinting (T1/T2 tiered), 
n-gram repetition, sentence opener diversity, punctuation 
fingerprinting, vocabulary clustering, density melody ensemble, 
Monte Carlo window sampling, and more. Weights were rebalanced 
after adversarial pen testing.

### Layer 2 — Forensic Character Scan
Scans raw Unicode codepoints for invisible characters 
(zero-width spaces, word joiners, soft hyphens), homoglyph 
substitutions (Cyrillic/Greek chars disguised as Latin), 
non-standard whitespace, and mixed quote encoding — artifacts 
that survive editing and are invisible to the naked eye.

### Layer 3 — File Metadata Analysis
Reads DOCX and PDF metadata directly in the browser — author 
fields, total editing time, revision session markers, 
created/modified timestamps, and application signatures. 
A document with 0 minutes of editing time tells a different 
story than one with 187.

### Layer 5 — Authorial Consistency Fingerprinting
Measures micro-habit consistency across five signals: contraction 
rate, Oxford comma usage, number formatting, sentence-final 
prepositions, and paragraph opener word class. Scores the 
*consistency* of each habit, not the habit itself — so formal 
and casual writers both score human for different reasons.

---

## Methodology

This tool was built iteratively using an adversarial approach:

1. Built initial 12-vector linguistic detector
2. Generated evasion text  
   specifically crafted to fool it
3. Analyzed exactly which signals were bypassed and why
4. Patched weaknesses and rebalanced weights
5. Added forensic and metadata layers targeting signals 
   that survive stylistic evasion

This mirrors the red-teaming process used by AI safety 
teams at companies like Anthropic and OpenAI.

---

## Tech

Pure HTML/CSS/JavaScript. No frameworks, no dependencies 
(PDF.js loaded from CDN for PDF parsing). 
Runs entirely client-side.

---

## Skills demonstrated

- NLP & computational linguistics
- Adversarial ML / red-teaming methodology  
- Digital forensics (Unicode, file metadata, binary parsing)
- Iterative system design from first principles
