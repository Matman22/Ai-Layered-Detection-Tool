/**
 * Feature extractor for the Top-1 stylometric classifier.
 *
 * Loads the REAL detection logic from index.html (same approach as
 * raid_eval/node_runner.js) and emits the 17-signal Layer-1 feature vector for
 * every labeled row in eval/real_dataset.csv. Because the features come from
 * index.html's own functions, the vectors used for TRAINING are byte-identical
 * to the ones the browser will compute at INFERENCE — no port drift.
 *
 * Output: eval/features.csv  (17 signal columns + is_ai_generated)
 * Usage:  node eval/extract_features.js
 */

const fs = require('fs');
const path = require('path');

// ─── Minimal DOM / browser stubs (copied from node_runner.js) ────────────────
const _el = () => ({
  addEventListener: () => {}, removeEventListener: () => {},
  classList: { add: () => {}, remove: () => {}, toggle: () => {}, contains: () => false },
  style: {}, innerHTML: '', textContent: '', value: '', disabled: false,
  scrollIntoView: () => {},
});
global.document = {
  getElementById: _el,
  createElement: () => ({ setAttribute: () => {}, appendChild: () => {}, src: '', onload: null, onerror: null }),
  head: { appendChild: () => {} },
};
global.window = { pdfjsLib: null };
global.alert = () => {};
global.TextEncoder = global.TextEncoder || require('util').TextEncoder;
global.TextDecoder = global.TextDecoder || require('util').TextDecoder;

// ─── Load & eval index.html's script block ───────────────────────────────────
const htmlPath = path.join(__dirname, '..', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) { process.stderr.write('ERROR: no <script> block in index.html\n'); process.exit(1); }
try { eval(scriptMatch[1]); }
catch (e) { process.stderr.write(`Eval warning (non-fatal): ${e.message}\n`); }

// ─── Feature names — order MUST match node_runner.js scores[0..16] ───────────
const FEATURE_NAMES = [
  'perplexity', 'burstiness', 'lexical', 'aiPhrases', 'hedging', 'passive',
  'transitions', 'clauseDepth', 'punctuation', 'paraUniformity', 'rareWords',
  'formality', 'ngramRep', 'openerDiv', 'punctFinger', 'vocabCluster', 'densityMelody',
];

function featureVector(text) {
  return [
    calcPerplexity(text).score,
    calcBurstiness(text),
    calcLexicalDiversity(text),
    calcAIPhrases(text).score,
    calcHedging(text),
    calcPassiveVoice(text),
    calcTransitions(text),
    calcClauseDepth(text),
    calcPunctuationVariance(text),
    calcParagraphUniformity(text),
    calcRareWords(text),
    calcFormalityShift(text),
    calcNgramRepetition(text),
    calcSentenceOpenerDiversity(text),
    calcPunctuationFingerprint(text),
    calcVocabClustering(text),
    calcDensityMelodyEnsemble(text).score,
  ];
}

// L2/L5 are NOT model features — they're emitted only so the trainer can
// reconstruct the true product verdict (L1*0.75 + L2*0.15 + L5*0.10) both ways.
function auxScores(text) {
  return [runForensicAnalysis(text).score, runAuthoralConsistency(text).score];
}

// ─── CSV parsing (handles quoted fields) ─────────────────────────────────────
function parseCSVRow(line) {
  const out = []; let cur = ''; let q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { if (q && line[i + 1] === '"') { cur += '"'; i++; } else q = !q; }
    else if (ch === ',' && !q) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

// ─── Main ────────────────────────────────────────────────────────────────────
const csvPath = path.join(__dirname, 'real_dataset.csv');
const raw = fs.readFileSync(csvPath, 'utf8').replace(/^﻿/, '');
const lines = raw.split('\n').filter(l => l.trim());
const headers = parseCSVRow(lines[0]);
const textIdx = headers.indexOf('text_content');
const labelIdx = headers.indexOf('is_ai_generated');
if (textIdx === -1 || labelIdx === -1) {
  process.stderr.write(`ERROR: expected text_content,is_ai_generated columns; got ${headers}\n`);
  process.exit(1);
}

const MIN_WORDS = 20; // mirror scoreText's insufficient-text guard
const outRows = [[...FEATURE_NAMES, 'is_ai_generated', 'l2_score', 'l5_score'].join(',')];
let kept = 0, skipped = 0;

for (let i = 1; i < lines.length; i++) {
  const cols = parseCSVRow(lines[i]);
  const text = (cols[textIdx] || '').trim();
  const label = cols[labelIdx];
  if (text.split(/\s+/).length < MIN_WORDS) { skipped++; continue; }
  try {
    const fv = featureVector(text);
    if (fv.some(v => typeof v !== 'number' || Number.isNaN(v))) { skipped++; continue; }
    const [l2, l5] = auxScores(text);
    outRows.push([...fv.map(v => v.toFixed(4)), label, l2.toFixed(4), l5.toFixed(4)].join(','));
    kept++;
  } catch (e) {
    process.stderr.write(`skip row ${i}: ${e.message}\n`);
    skipped++;
  }
  if (i % 50 === 0) process.stdout.write(`  ${i}/${lines.length - 1}\r`);
}

const outPath = path.join(__dirname, 'features.csv');
fs.writeFileSync(outPath, outRows.join('\n'), 'utf8');
console.log(`\nWrote ${kept} feature rows (skipped ${skipped}) to ${outPath}`);
