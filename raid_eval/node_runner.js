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

  // Weights — must match the tuned array in index.html
  const weights = [
    0.01, // perplexity
    0.11, // burstiness
    0.06, // lexical diversity
    0.07, // AI phrases
    0.01, // hedging
    0.01, // passive voice
    0.03, // transitions
    0.03, // clause depth
    0.07, // punctuation variance
    0.12, // paragraph uniformity
    0.02, // rare words
    0.12, // register stability
    0.04, // n-gram repetition
    0.05, // sentence opener diversity
    0.07, // punctuation fingerprint
    0.07, // vocab clustering
    0.07, // density melody
    0.04, // monte carlo
  ];

  const scores = [
    perplexity.score, burstiness, lexical, aiPhrases, hedging, passive,
    transitions, clauseDepth, punctuation, paraUniformity, rareWords, formality,
    ngramRep, openerDiv, punctFinger, vocabCluster, densityMelody, monteCarlo.mean,
  ];

  // Evidence accumulation (only signals above 50 contribute)
  const evidence     = scores.reduce((sum, s, i) => sum + Math.max(0, s - 50) * weights[i], 0);
  const baseComposite = Math.min(100, Math.round(evidence * 5));

  // Convergence bonus: anchor signals clustering above 70
  const anchorIdx  = [1, 3, 11, 15];
  const anchorsHot = anchorIdx.filter(i => scores[i] > 70).length;
  const convBonus  = [0, 0, 5, 12, 18][Math.min(anchorsHot, 4)];

  // Smoking-gun bonus: saturated AI phrase score
  const smokingGunBonus = scores[3] >= 100 ? 30 : scores[3] > 85 ? 15 : 0;

  const composite    = Math.min(100, baseComposite + convBonus + smokingGunBonus);
  const combinedScore = Math.round(Math.min(100, composite * 0.75 + forensic.score * 0.15 + layer5.score * 0.10));
  const trueCombined  = combinedScore; // no metadata for plain-text samples

  let verdict;
  if      (trueCombined >= 50) verdict = 'AI';
  else if (trueCombined >= 20) verdict = 'Mixed';
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
