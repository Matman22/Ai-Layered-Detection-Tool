/**
 * Iteration 1 experiment: 17 base stylometric features + 4 readability candidates.
 *
 * Base features come from eval'ing index.html's <script> block with DOM stubs —
 * copied verbatim from eval/extract_features.js so the 17 base columns stay
 * byte-identical. The 4 candidates (fleschKincaid, gunningFog,
 * functionWordRatio, commaRate) are computed in plain JS from the raw text.
 *
 * Output: eval/experiments/features_ext.csv
 * Usage:  node eval/experiments/extract_ext.js
 */

const fs = require('fs');
const path = require('path');

// ─── Minimal DOM / browser stubs (copied from extract_features.js) ───────────
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
const htmlPath = path.join(__dirname, '..', '..', 'index.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) { process.stderr.write('ERROR: no <script> block in index.html\n'); process.exit(1); }
try { eval(scriptMatch[1]); }
catch (e) { process.stderr.write(`Eval warning (non-fatal): ${e.message}\n`); }

// ─── Feature names — order MUST match extract_features.js ────────────────────
const FEATURE_NAMES = [
  'perplexity', 'burstiness', 'lexical', 'aiPhrases', 'hedging', 'passive',
  'transitions', 'clauseDepth', 'punctuation', 'paraUniformity', 'rareWords',
  'formality', 'ngramRep', 'openerDiv', 'punctFinger', 'vocabCluster', 'densityMelody',
];
const EXT_NAMES = ['fleschKincaid', 'gunningFog', 'functionWordRatio', 'commaRate'];

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

// ─── 4 readability candidates (plain JS, raw text) ───────────────────────────
const STOPWORDS = new Set([
  'the','a','an','and','or','but','if','of','to','in','on','at','by','for','with',
  'about','as','into','through','after','is','are','was','were','be','been','being',
  'have','has','had','do','does','did','will','would','can','could','shall','should',
  'may','might','must','that','which','who','whom','this','these','those','it','its',
  'he','she','they','them','his','her','their','we','us','our','you','your','i','me',
  'my','not','no','than','then','so','too','very',
]);

function estSyllables(word) {
  const groups = word.toLowerCase().match(/[aeiouy]+/g);
  return Math.max(1, groups ? groups.length : 0);
}

function extFeatures(text) {
  const words = text.split(/\s+/).filter(w => w.length > 0);
  const nWords = Math.max(1, words.length);
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const nSents = Math.max(1, sentences.length);

  let sylTotal = 0, complexWords = 0;
  for (const w of words) {
    const s = estSyllables(w);
    sylTotal += s;
    if (s >= 3) complexWords++;
  }

  const fleschKincaid = 206.835 - 1.015 * (nWords / nSents) - 84.6 * (sylTotal / nWords);
  const gunningFog = 0.4 * ((nWords / nSents) + 100 * complexWords / nWords);

  const tokens = words.map(w => w.toLowerCase().replace(/[^a-z']/g, '')).filter(t => t.length > 0);
  const stopCount = tokens.reduce((s, t) => s + (STOPWORDS.has(t) ? 1 : 0), 0);
  const functionWordRatio = tokens.length ? stopCount / tokens.length : 0;

  const commas = (text.match(/,/g) || []).length;
  const commaRate = 100 * commas / nWords;

  return [fleschKincaid, gunningFog, functionWordRatio, commaRate];
}

// ─── CSV parsing (copied from extract_features.js) ───────────────────────────
// Full-file CSV parser — respects quoted fields that span multiple newlines.
function parseCSV(text) {
  const records = [];
  let field = '', record = [], q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; } else q = false;
      } else field += ch;
    } else if (ch === '"') { q = true; }
    else if (ch === ',') { record.push(field); field = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      record.push(field); field = '';
      if (record.length > 1 || record[0] !== '') records.push(record);
      record = [];
    } else field += ch;
  }
  if (field !== '' || record.length) { record.push(field); records.push(record); }
  return records;
}

// ─── Main ────────────────────────────────────────────────────────────────────
const csvPath = path.join(__dirname, '..', 'combined_dataset.csv');
const raw = fs.readFileSync(csvPath, 'utf8').replace(/^﻿/, '');
const records = parseCSV(raw);
const headers = records[0];
const textIdx = headers.indexOf('text_content');
const labelIdx = headers.indexOf('is_ai_generated');
const sourceIdx = headers.indexOf('source');
if (textIdx === -1 || labelIdx === -1 || sourceIdx === -1) {
  process.stderr.write(`ERROR: expected text_content,is_ai_generated,source columns; got ${headers}\n`);
  process.exit(1);
}

const MIN_WORDS = 20; // mirror scoreText's insufficient-text guard
const outHeader = [...FEATURE_NAMES, ...EXT_NAMES, 'is_ai_generated', 'source'];
const outRows = [outHeader.join(',')];
let kept = 0, skipped = 0;

for (let i = 1; i < records.length; i++) {
  const cols = records[i];
  const text = (cols[textIdx] || '').trim();
  const label = cols[labelIdx];
  if (text.split(/\s+/).length < MIN_WORDS) { skipped++; continue; }
  try {
    const fv = featureVector(text);
    if (fv.some(v => typeof v !== 'number' || Number.isNaN(v))) { skipped++; continue; }
    const ext = extFeatures(text);
    if (ext.some(v => typeof v !== 'number' || Number.isNaN(v))) { skipped++; continue; }
    const row = [...fv.map(v => v.toFixed(4)), ...ext.map(v => v.toFixed(4)), label, cols[sourceIdx] || ''];
    outRows.push(row.join(','));
    kept++;
  } catch (e) {
    process.stderr.write(`skip row ${i}: ${e.message}\n`);
    skipped++;
  }
  if (i % 50 === 0) process.stdout.write(`  ${i}/${records.length - 1}\r`);
}

const outPath = path.join(__dirname, 'features_ext.csv');
fs.writeFileSync(outPath, outRows.join('\n'), 'utf8');
console.log(`\nWrote ${kept} feature rows (skipped ${skipped}) to ${outPath}`);
