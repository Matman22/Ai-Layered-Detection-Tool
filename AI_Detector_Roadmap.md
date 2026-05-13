# AI Origin Detector — Feature Roadmap & Industry Report
*Generated April 2026*

---

## Part 1 — What's Happening in the Industry Right Now

### The State of AI Detection in 2026

The field is moving fast and the honest picture is messier than the marketing suggests.
Commercial tools like Originality.ai and GPTZero claim 98–99% accuracy, but those
numbers come from controlled benchmarks on raw, unedited AI text. Independent research
tells a different story — real-world accuracy on edited or mixed-origin content
consistently falls in the 40–80% range, and false positive rates on ESL writers
have hit as high as 61% in peer-reviewed studies.

As a direct result, a growing list of elite universities have disabled AI detection
entirely — Vanderbilt, Yale, Johns Hopkins, Northwestern, UCLA, UC San Diego, UT Austin,
Michigan State, and 10+ others. The reason is consistent: the tools are not reliable
enough to use as evidence in academic misconduct proceedings.

The field is splitting into two camps as a result:

**Camp 1 — Better detectors.** More training data, ensemble models, new techniques
like rewriting-based detection (see RAIDAR below). Ongoing arms race.

**Camp 2 — Provenance and disclosure.** Rather than detecting AI after the fact,
embed proof of human authorship at the point of creation. Cryptographic signing,
keystroke logging, process verification. This is where serious institutional money
is going in 2026.

---

### The Most Important New Research: RAIDAR

The most significant academic breakthrough in detection methodology right now is
**RAIDAR** (geneRative AI Detection viA Rewriting), published at ICLR 2024 by
Columbia Engineering researchers.

The core insight is elegant: LLMs treat AI-generated text as already high quality,
so when asked to rewrite it they make very few changes. When asked to rewrite
human text, they make many more changes because human writing has more "room for
improvement" by AI standards.

RAIDAR works by:
1. Taking the input text
2. Sending it to an LLM with a rewriting prompt
3. Measuring the edit distance between original and rewritten version
4. Small distance = AI-generated. Large distance = human-written.

It improved F1 detection scores by up to 29 points over previous methods across
news, essays, creative writing, code, and reviews. Crucially it works without
access to the original generating model — you can use GPT to detect Claude output
and vice versa.

**This is directly implementable in our tool** using the Anthropic API artifact
capability. It would be our single most impactful upgrade.

---

### Watermarking: SynthID and the C2PA Standard

**Google's SynthID** embeds invisible watermarks directly into token probability
distributions at generation time — not in the text itself, but in the statistical
pattern of which words were chosen. It has already watermarked over 10 billion
pieces of content. Critical limitation: it only detects Google AI content. It
cannot identify ChatGPT, Claude, or any non-Google output.

**C2PA (Coalition for Content Provenance and Authenticity)** is an industry standard
backed by Adobe, Microsoft, Google, and the BBC that attaches cryptographically
signed provenance records to files — who created them, when, with what tool.
California's AI Transparency Act (effective January 2026) requires "latent
disclosure" markers in AI-generated content. The IETF proposed an "AI Content
Disclosure Header" web standard in September 2025.

The direction is clear: the future of AI detection is **provenance at creation**,
not analysis after the fact. Our tool should be watching this space.

---

### GPTZero's Source Finder

GPTZero added a feature in 2025 called Source Finder that specifically catches
**fabricated citations** — a signature AI failure mode. AI confidently cites
papers, people, and studies that don't exist. Checking whether the citations in
a document are real is a powerful orthogonal signal that doesn't depend on
linguistic analysis at all. This is something our tool could implement.

---

### Turnitin's Bypasser Detection (August 2025)

Turnitin added detection specifically targeting text that was AI-generated then
run through "humanizer" tools. It works by looking for characteristic artifacts
that humanizers leave behind — the statistical fingerprint of the humanizing
process itself rather than the original AI text. Independent testing shows
accuracy drops to 20–63% on sophisticated humanized text, but the direction
is important: detecting the *process* of evasion, not just the output.

---

## Part 2 — Feature Roadmap

Features are organized by priority tier. Tiers are based on impact,
implementability with current tech stack, and how much they advance the
project from a portfolio/research perspective.

---

## TIER 1 — High Impact, Implement Next

---

### Feature 1: RAIDAR-Style Rewriting Analysis (Layer 4)
**What it is:** Send the pasted text to an LLM via the Anthropic API (already
available in our artifact setup), ask it to rewrite the text, then measure
how much it changed using edit distance scoring. Small changes = AI signal.
Large changes = human signal.

**Why it matters:** This is the current state-of-the-art academic technique.
It adds a completely different category of signal — one that doesn't depend
on linguistic rules at all, but on how an LLM itself perceives the text.
It would push our accuracy significantly closer to commercial tools.

**How to build it:** The artifact API capability is already wired. Send the
text with a prompt like "Rewrite this passage to improve clarity and flow."
Compare word-level or character-level edit distance between original and
returned text. Score it.

**Difficulty:** Medium. API calls are already supported in the artifact system.

---

### Feature 2: Semantic Coherence Mapping (Layer 4 alternate)
**What it is:** Map how ideas travel across the document. Measure
paragraph-to-paragraph lexical overlap, callback density (do words from
paragraph 1 reappear in paragraph 5?), and conclusion vocabulary novelty.
AI flows forward smoothly and evenly. Humans jump, circle back, and
introduce new framing at the end.

**Why it matters:** This targets exactly what the marine biology pen test
evaded. A text can have perfect burstiness and no AI phrases but still
have the unmistakable smooth semantic flow of machine generation.

**How to build it:** Pure JavaScript. Extract content word sets per paragraph,
compute Jaccard similarity between non-adjacent paragraphs, measure callback
rate, score conclusion novelty. No external dependencies.

**Difficulty:** Medium. All client-side.

---

### Feature 3: Authorial Consistency Fingerprinting (Layer 5) ✅ COMPLETED
**What it is:** Measure micro-habit consistency across the document —
contraction usage (don't vs do not), Oxford comma consistency, number
formatting (3 vs three), sentence-final preposition tolerance, paragraph
opening word class. Score the *consistency of the habit* not the habit
itself.

**Why it matters:** Humans are consistent about their stylistic choices
because they're habits. AI varies these based on local context, producing
detectable inconsistency. This doesn't penalize any particular style —
a formal writer and a casual writer both score human for different reasons.

**Status:** Fully implemented as Layer 5 — Authorial Consistency Fingerprinting.
Contributes 10% of the final combined score. Five weighted micro-habit checks,
each with a dedicated UI panel. Scores consistency of the habit, not the habit itself.

---

### Feature 4: Citation Verification
**What it is:** Extract any citations, references, book titles, paper
names, or named studies from the text. Run a web search against them
to verify they exist. Flag fabricated citations.

**Why it matters:** AI hallucinating citations is one of its most reliable
failure modes. A document confidently citing three papers that don't exist
is a very strong AI signal that no linguistic analysis catches. GPTZero
added this feature in 2025 and it's considered one of their most useful additions.

**How to build it:** Regex to extract citation patterns. Web search tool
already available in the artifact API setup. Cross-reference results.

**Difficulty:** Medium. Requires API integration but the infrastructure
is already available.

---

## TIER 2 — Strong Additions, Build After Tier 1

---

### Feature 5: Sentence-Level Heatmap Visualization
**What it is:** Instead of just giving an overall score, highlight
individual sentences in the text with a color gradient — red for
high AI confidence, green for human. Show the user exactly *which*
sentences triggered which signals.

**Why it matters:** This is what GPTZero's sentence-level highlighting
does and it's one of the most-cited reasons users trust it more than
tools that just give a percentage. It transforms the tool from a
black box into something explainable.

**How to build it:** Run per-sentence scoring on the key metrics —
phrase fingerprinting, n-gram repetition, opener patterns. Render
the text with CSS background color gradients mapped to per-sentence
scores.

**Difficulty:** Medium. UI work more than algorithmic work.

---

### Feature 6: Multi-Model Comparison Mode
**What it is:** Run the text through two or three different AI models
via the Anthropic API (using the RAIDAR technique) and show how each
model responds to it. If GPT-style output gets barely changed by
Claude but heavily changed by itself — that's a signal.

**Why it matters:** Different models have different biases about what
counts as "high quality." Using multiple rewriters gives you a more
robust signal and reduces false positives.

**Difficulty:** Medium-High. Multiple API calls, result synthesis.

---

### Feature 7: Document History Timeline (DOCX)
**What it is:** For DOCX files, visualize the revision history as a
timeline — when the document was created, how many sessions, total
edit time, when tracked changes were made, who made them. Display
it graphically rather than just as numbers.

**Why it matters:** The metadata layer is currently our most unique
feature vs commercial tools. Making it visual and compelling
dramatically increases its impact for portfolio purposes and
makes it genuinely more useful.

**Difficulty:** Low. The data is already being extracted — it's a
UI/visualization task.

---

### Feature 8: Paste History Detection
**What it is:** Use the browser's input event timing to measure
whether text arrived character-by-character (typed) or all at once
(pasted). A massive chunk of text arriving in a single input event
is a strong signal that it wasn't typed in the textarea directly.

**Why it matters:** Most AI text gets pasted into documents. Detecting
the paste event itself — not just the text content — is an orthogonal
signal that works regardless of how good the writing looks.

**How to build it:** Track input event timing and character delta size.
If 500+ characters arrive in under 50ms it's a paste. Log and flag.

**Difficulty:** Low. Pure JavaScript event handling.

---

### Feature 9: Readability Consistency Scoring
**What it is:** Run Flesch-Kincaid readability scoring per paragraph
and measure how consistent it is across the document. AI maintains
suspiciously consistent readability scores throughout. Humans naturally
vary their complexity as they work through an argument.

**Why it matters:** Another signal that catches stylistic evasion.
You can vary sentence length and still maintain perfectly consistent
readability if you're an AI — the two signals are independent.

**Difficulty:** Low. Flesch-Kincaid is a simple formula implementable
in JavaScript.

---

## TIER 3 — Ambitious, Longer-Term

---

### Feature 10: Trained Classifier Integration
**What it is:** Move from rule-based scoring to an actual trained
ML classifier. Collect labeled samples of human vs AI text, extract
the 16 features already computed by the tool, train a logistic
regression or small neural net on them using scikit-learn in Python.
Export the model weights and run inference in JavaScript.

**Why it matters:** This is the architectural leap that takes the
tool from sophisticated rules to actual machine learning. It's also
a genuine research project — collecting and labeling training data,
running experiments, analyzing results. Directly publishable at
undergraduate level.

**Difficulty:** High. Requires data collection, Python ML pipeline,
model export. Semester-scale project.

---

### Feature 11: C2PA Provenance Header Reading
**What it is:** When a file is dropped in, check for C2PA provenance
metadata — cryptographically signed records of how the document was
created, by what tool, at what time. As this standard becomes more
widely adopted, documents created with AI tools will carry these
markers. Read and display them.

**Why it matters:** This is where the industry is heading. Being
ahead of the curve on provenance standards is a strong research
positioning angle.

**Difficulty:** High. C2PA is a complex standard. JavaScript libraries
are emerging but still maturing.

---

### Feature 12: Browser Extension Version
**What it is:** Package the detection logic as a Chrome or Firefox
extension that can analyze any text on any webpage — highlight
AI-generated passages inline as you browse, right-click to analyze
selected text.

**Why it matters:** Dramatically increases the tool's real-world
usefulness and reach. A browser extension is a much more compelling
portfolio item than a standalone HTML file.

**Difficulty:** High. Extension packaging, permissions, content
script architecture. But all the detection logic already exists.

---

### Feature 13: API Mode
**What it is:** Add a mode where the tool exposes a simple REST API
that other applications can call — send text, get back a JSON score
report. Could be hosted cheaply on Vercel or Cloudflare Workers.

**Why it matters:** Transforms the project from a tool into a
platform. Other developers could integrate your detection into
their own applications. This is how GPTZero grew — API access to
researchers drove adoption and validation.

**Difficulty:** High. Requires server-side hosting and API design.
But the detection logic is all client-side already.

---

## Part 3 — What's Currently Built (Reference)

### Layer 1 — Linguistic Analysis (18 inputs)
- Perplexity ensemble (3-proxy: word-length variance, syllable density, trigram entropy)
- Sentence burstiness (length CV)
- Lexical diversity (TTR)
- AI phrase fingerprinting (T1: ~28 near-exclusive phrases; T2: ~33 common filler; tiered scoring)
- Hedging language density
- Passive voice rate
- Transition word uniformity
- Clause depth index
- Punctuation variance
- Paragraph uniformity
- Rare word profiling
- Register stability / formality shift
- N-gram repetition (3-word sequences)
- Sentence opener diversity
- Punctuation fingerprinting (em-dash, parenthetical)
- Vocabulary clustering
- **Density Melody Ensemble** (5 independent runs at varying window/granularity, confidence-weighted)
- **Monte Carlo Window Sampling** (8–20 random windows, 5 signals per window, variance = mixed-origin signal)

### Layer 2 — Forensic Character Scan
- 23 invisible Unicode character types
- 32 Cyrillic/Greek homoglyph substitutions
- Smart/straight quote mixing detection
- Em-dash and ellipsis style analysis
- Non-standard whitespace profiling
- Foreign script block mapping

### Layer 3 — File Metadata Analysis (DOCX + PDF)
- Author field validation
- Total editing time (Word's TotalTime field)
- Created vs modified timestamp delta
- Revision session marker count (rsid)
- Tracked changes (insertions/deletions)
- Application/producer fingerprinting
- Save count (revision number)
- PDF metadata extraction via PDF.js

### Layer 5 — Authorial Consistency Fingerprinting
- Contraction rate consistency across paragraphs
- Oxford comma consistency
- Number formatting consistency (words vs digits)
- Sentence-final preposition consistency
- Paragraph opener word-class consistency

### Infrastructure
- Drag-and-drop file upload (TXT, DOCX, PDF, MD, RTF)
- PDF.js integration for robust PDF parsing
- Five-layer combined scoring system
- Per-signal flag display with severity levels
- Monte Carlo tab with positional heatmap and classification badge
- Full analysis log
- Runs 100% client-side — nothing sent anywhere

---

*This document should be updated each time a new feature is implemented.*
*Last updated: May 2026*
