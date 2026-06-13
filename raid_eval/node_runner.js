/**
 * Node.js wrapper that loads the detection logic from index.html
 * and scores text without any browser/DOM dependencies.
 *
 * Reads a JSON array of {id, text} from stdin.
 * Writes a JSON array of {id, score, verdict, l1, l2, l5} to stdout.
 */

const fs = require('fs');
const path = require('path');

// ─── Minimal DOM / browser stubs ────────────────────────────────────────────
// index.html's script block runs top-level DOM code on load; stub it out.
const _el = () => ({
  addEventListener: () => {},
  removeEventListener: () => {},
  classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
  style: {},
  innerHTML: '',
  textContent: '',
  value: '',
  disabled: false,
  scrollIntoView: () => {},
});

global.document = {
  getElementById: _el,
  createElement: () => ({ setAttribute: () => {}, appendChild: () => {}, src: '', onload: null, onerror: null }),
  head: { appendChild: () => {} },
};
global.window = { pdfjsLib: null };
global.alert = () => {};
// TextEncoder/TextDecoder are available in Node >= 11 but need to be global for the script
global.TextEncoder = global.TextEncoder || require('util').TextEncoder;
global.TextDecoder = global.TextDecoder || require('util').TextDecoder;

// ─── Load & eval the detection logic from index.html ────────────────────────
const htmlPath = path.join(__dirname, '..', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) { process.stderr.write('ERROR: No <script> block found in index.html\n'); process.exit(1); }

// Wrap in a function scope to avoid leaking 'const' declarations into global
// (Node throws on re-declaration if eval'd at top level twice)
const scriptSrc = scriptMatch[1];
try {
  eval(scriptSrc);
} catch (e) {
  // Some top-level DOM accesses may still throw — log and continue.
  // The detection functions are defined before any DOM errors occur.
  process.stderr.write(`Eval warning (non-fatal): ${e.message}\n`);
}

// ─── Core scorer (mirrors runAnalysis() logic without DOM) ──────────────────
function scoreText(text) {
  if (!text || text.trim().split(/\s+/).length < 20) {
    return { score: 50, verdict: 'Insufficient', l1: 50, l2: 0, l5: 50 };
  }

  const perplexity       = calcPerplexity(text);
  const burstiness       = calcBurstiness(text);
  const lexical          = calcLexicalDiversity(text);
  const { score: aiPhrases } = calcAIPhrases(text);
  const hedging          = calcHedging(text);
  const passive          = calcPassiveVoice(text);
  const transitions      = calcTransitions(text);
  const clauseDepth      = calcClauseDepth(text);
  const punctuation      = calcPunctuationVariance(text);
  const paraUniformity   = calcParagraphUniformity(text);
  const rareWords        = calcRareWords(text);
  const formality        = calcFormalityShift(text);
  const ngramRep         = calcNgramRepetition(text);
  const openerDiv        = calcSentenceOpenerDiversity(text);
  const punctFinger      = calcPunctuationFingerprint(text);
  const vocabCluster     = calcVocabClustering(text);
  const densityMelody    = calcDensityMelodyEnsemble(text).score;
  const monteCarlo       = runMonteCarloAnalysis(text);
  const forensic         = runForensicAnalysis(text);
  const layer5           = runAuthoralConsistency(text);

  const scores = [
    perplexity.score, burstiness, lexical, aiPhrases, hedging, passive,
    transitions, clauseDepth, punctuation, paraUniformity, rareWords, formality,
    ngramRep, openerDiv, punctFinger, vocabCluster, densityMelody,
  ];

  // Layer-1 score from the trained classifier (classifierL1 is a function
  // declaration in index.html — leaks from eval, stays in sync automatically).
  const composite = Math.round(classifierL1(scores));

  // Phase 3 learned model — must match the ML_WEIGHTS array in index.html.
  // const declarations don't leak from eval, so this is a local copy.
  const ML_WEIGHTS = [
    +0.002839, +0.046114, +0.029001, +0.054033, -0.022433, -0.020644, -0.073171,
    -0.001024, +0.017217, +0.149378, -0.061777, +0.004403, +0.061725, -0.001617,
    -0.029902, -0.089321, +0.021054, +0.078046, +0.035661, -0.028986,
  ];
  const ML_INTERCEPT = -10.582955;
  const mlFeatures = [
    perplexity.score, burstiness, lexical, aiPhrases, hedging, passive, transitions,
    clauseDepth, punctuation, paraUniformity, rareWords, formality, ngramRep, openerDiv,
    punctFinger, vocabCluster, densityMelody, monteCarlo.mean, forensic.score, layer5.score,
  ];
  const mlLogit = mlFeatures.reduce((s, x, i) => s + x * ML_WEIGHTS[i], ML_INTERCEPT);
  const combinedScore = Math.round(100 / (1 + Math.exp(-mlLogit)));
  const trueCombined  = combinedScore; // no metadata for plain-text samples

  let verdict;
  if      (trueCombined >= 60) verdict = 'AI';
  else if (trueCombined >= 40) verdict = 'Mixed';
  else                          verdict = 'Human';

  return { score: trueCombined, verdict, l1: composite, l2: forensic.score, l5: layer5.score };
}

// ─── Batch processing via stdin/stdout ──────────────────────────────────────
let raw = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { raw += chunk; });
process.stdin.on('end', () => {
  // Strip UTF-8 BOM (U+FEFF) that PowerShell sometimes injects
  const cleaned = raw.replace(/^﻿/, '');
  let items;
  try { items = JSON.parse(cleaned); } catch (e) {
    process.stderr.write(`JSON parse error: ${e.message}\n`);
    process.exit(1);
  }

  const results = items.map(item => {
    try {
      return { id: item.id, ...scoreText(item.text) };
    } catch (e) {
      process.stderr.write(`Score error on id=${item.id}: ${e.message}\n`);
      return { id: item.id, score: 50, verdict: 'Error', l1: 0, l2: 0, l5: 0 };
    }
  });

  process.stdout.write(JSON.stringify(results));
});
