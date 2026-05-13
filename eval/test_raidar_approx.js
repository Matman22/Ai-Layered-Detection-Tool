/**
 * RAIDAR Approximation — Eval Script
 *
 * Tests 4 local signals that approximate what RAIDAR detects without an API:
 *
 * 1. Inter-sentence similarity — AI sentences are more uniformly similar to each other
 * 2. Self-BLEU — AI sentences share more bigrams with the rest of the document
 * 3. Vocabulary predictability — AI uses consistent vocabulary across document halves
 * 4. Entropy uniformity — AI sentences have uniform word-frequency entropy
 *
 * Reports each signal's individual discrimination power, then tests a combined
 * Layer 4 score vs the existing 16-vector baseline.
 */

const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join(__dirname, 'real_dataset.csv');

// ─── CSV parser ───────────────────────────────────────────────────────────────
function parseCSV(raw) {
  const lines = raw.split('\n').filter(l => l.trim());
  const headers = parseCSVRow(lines[0]);
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVRow(lines[i]);
    if (values.length < headers.length) continue;
    const row = {};
    headers.forEach((h, idx) => { row[h.trim()] = values[idx]; });
    rows.push(row);
  }
  return rows;
}

function parseCSVRow(line) {
  const result = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') { current += '"'; i++; }
      else { inQuotes = !inQuotes; }
    } else if (ch === ',' && !inQuotes) {
      result.push(current); current = '';
    } else { current += ch; }
  }
  result.push(current);
  return result;
}

// ─── Text utilities ───────────────────────────────────────────────────────────
function getSentences(text) {
  return (text.match(/[^.!?]+[.!?]+/g) || [text]).filter(s => s.trim().length > 15);
}

// ─── Signal 1: Inter-sentence Jaccard similarity ──────────────────────────────
// AI sentences are uniformly similar to each other (same distribution).
// Human sentences vary more — some share a lot, others almost nothing.
function calcInterSentenceSimilarity(text) {
  const sentences = getSentences(text);
  if (sentences.length < 4) return 50;

  const sets = sentences.map(s =>
    new Set((s.toLowerCase().match(/\b[a-z]{3,}\b/g) || []))
  );

  const similarities = [];
  for (let i = 0; i < sets.length - 1; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      const inter = [...sets[i]].filter(w => sets[j].has(w)).length;
      const union = new Set([...sets[i], ...sets[j]]).size;
      if (union > 0) similarities.push(inter / union);
    }
  }
  if (!similarities.length) return 50;

  const mean = similarities.reduce((a, b) => a + b, 0) / similarities.length;
  const variance = similarities.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / similarities.length;
  const stdDev = Math.sqrt(variance);

  // High mean + low stdDev = uniform similarity = AI
  const meanSignal = Math.min(100, mean * 400);
  const uniformitySignal = Math.max(0, 100 - stdDev * 600);
  return Math.round(meanSignal * 0.55 + uniformitySignal * 0.45);
}

// ─── Signal 2: Self-BLEU (bigram overlap between sentences) ──────────────────
// AI sentences share more bigrams with the rest of the document.
// Each sentence's bigrams appear repeatedly across the document in AI text.
function calcSelfBLEU(text) {
  const sentences = getSentences(text);
  if (sentences.length < 3) return 50;

  const tokenized = sentences.map(s => s.toLowerCase().match(/\b[a-z]+\b/g) || []);

  let totalScore = 0;
  let counted = 0;
  for (let i = 0; i < tokenized.length; i++) {
    const thisBigrams = new Set();
    for (let k = 0; k < tokenized[i].length - 1; k++) {
      thisBigrams.add(tokenized[i][k] + ' ' + tokenized[i][k + 1]);
    }
    if (thisBigrams.size === 0) continue;

    const otherBigrams = new Set();
    for (let j = 0; j < tokenized.length; j++) {
      if (j === i) continue;
      for (let k = 0; k < tokenized[j].length - 1; k++) {
        otherBigrams.add(tokenized[j][k] + ' ' + tokenized[j][k + 1]);
      }
    }

    const overlap = [...thisBigrams].filter(bg => otherBigrams.has(bg)).length;
    totalScore += overlap / thisBigrams.size;
    counted++;
  }

  const avgSelfBLEU = counted > 0 ? totalScore / counted : 0;
  return Math.min(100, Math.round(avgSelfBLEU * 220));
}

// ─── Signal 3: Vocabulary predictability (split-half Jaccard) ────────────────
// Split document in half by words. High vocabulary overlap between halves = AI.
// AI uses a consistent vocabulary throughout; humans introduce new terms as they develop ideas.
function calcVocabPredictability(text) {
  const words = text.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
  if (words.length < 40) return 50;

  const mid = Math.floor(words.length / 2);
  const first = new Set(words.slice(0, mid));
  const second = new Set(words.slice(mid));

  const inter = [...first].filter(w => second.has(w)).length;
  const union = new Set([...first, ...second]).size;
  const jaccard = union > 0 ? inter / union : 0;

  return Math.min(100, Math.round(jaccard * 130));
}

// ─── Signal 4: Sentence entropy uniformity ────────────────────────────────────
// Compute word-frequency entropy for each sentence using document-wide frequencies.
// AI: each sentence has similar entropy (uniform predictability).
// Human: entropy varies — some sentences use rare words, others use common ones.
function calcEntropyUniformity(text) {
  const sentences = getSentences(text);
  if (sentences.length < 4) return 50;

  const allWords = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (allWords.length < 30) return 50;

  const freq = {};
  allWords.forEach(w => { freq[w] = (freq[w] || 0) + 1; });
  const total = allWords.length;

  const entropies = sentences.map(s => {
    const sw = s.toLowerCase().match(/\b[a-z]+\b/g) || [];
    if (!sw.length) return null;
    const h = -sw.reduce((sum, w) => {
      const p = (freq[w] || 1) / total;
      return sum + p * Math.log2(p);
    }, 0) / sw.length;
    return h;
  }).filter(e => e !== null);

  if (entropies.length < 3) return 50;

  const mean = entropies.reduce((a, b) => a + b, 0) / entropies.length;
  const variance = entropies.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / entropies.length;

  // Low variance = uniform entropy per sentence = AI
  return Math.max(0, Math.min(100, Math.round(85 - Math.sqrt(variance) * 280)));
}

// ─── Combined Layer 4 score ───────────────────────────────────────────────────
function runRAIDARApprox(text) {
  const interSentence = calcInterSentenceSimilarity(text);
  const selfBLEU      = calcSelfBLEU(text);
  const vocabPredict  = calcVocabPredictability(text);
  const entropyUnif   = calcEntropyUniformity(text);

  const score = Math.round(
    interSentence * 0.35 +
    selfBLEU      * 0.25 +
    vocabPredict  * 0.25 +
    entropyUnif   * 0.15
  );

  return { score, interSentence, selfBLEU, vocabPredict, entropyUnif };
}

// ─── Evaluation ───────────────────────────────────────────────────────────────
const rows = parseCSV(fs.readFileSync(CSV_PATH, 'utf8'));
console.log(`Evaluating RAIDAR approximation on ${rows.length} rows...\n`);

const aiResults   = [];
const humanResults = [];

rows.forEach((row, idx) => {
  const text = row['text_content'] || '';
  const isAI = parseInt(row['is_ai_generated']) === 1;
  const r = runRAIDARApprox(text);
  (isAI ? aiResults : humanResults).push(r);
  if ((idx + 1) % 100 === 0) process.stdout.write(`  ${idx + 1}/${rows.length}\r`);
});

// ─── Per-signal discrimination analysis ──────────────────────────────────────
const signals = [
  { key: 'score',          label: 'Combined Layer 4' },
  { key: 'interSentence',  label: 'Inter-sentence similarity' },
  { key: 'selfBLEU',       label: 'Self-BLEU' },
  { key: 'vocabPredict',   label: 'Vocab predictability' },
  { key: 'entropyUnif',    label: 'Entropy uniformity' },
];

console.log('\n╔═══════════════════════════════════════════════════════════╗');
console.log('║          RAIDAR APPROXIMATION — SIGNAL ANALYSIS          ║');
console.log('╠═══════════════════════════════════════════════════════════╣');
console.log(`║ ${'Signal'.padEnd(28)} ${'AI mean'.padEnd(9)} ${'Human mean'.padEnd(11)} ${'Gap'.padEnd(6)} ║`);
console.log('╠═══════════════════════════════════════════════════════════╣');

const mean = arr => (arr.reduce((a, b) => a + b, 0) / arr.length).toFixed(1);

signals.forEach(({ key, label }) => {
  const aiMean    = mean(aiResults.map(r => r[key]));
  const humanMean = mean(humanResults.map(r => r[key]));
  const gap       = (parseFloat(aiMean) - parseFloat(humanMean)).toFixed(1);
  const marker    = Math.abs(parseFloat(gap)) >= 5 ? ' ←' : '';
  console.log(`║ ${label.padEnd(28)} ${aiMean.padEnd(9)} ${humanMean.padEnd(11)} ${(gap + marker).padEnd(8)}║`);
});

console.log('╚═══════════════════════════════════════════════════════════╝');

// ─── Threshold sweep on combined Layer 4 score ───────────────────────────────
console.log('\nThreshold sweep (Layer 4 combined score):');
let bestF1 = 0, bestT = 50, bestStats = {};

for (let t = 20; t <= 80; t += 5) {
  const tp = aiResults.filter(r => r.score >= t).length;
  const tn = humanResults.filter(r => r.score < t).length;
  const fp = humanResults.filter(r => r.score >= t).length;
  const fn = aiResults.filter(r => r.score < t).length;
  const total = tp + tn + fp + fn;
  const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const rec  = tp + fn > 0 ? tp / (tp + fn) : 0;
  const f1   = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
  const acc  = (tp + tn) / total;
  if (f1 > bestF1) { bestF1 = f1; bestT = t; bestStats = { tp, tn, fp, fn, prec, rec, f1, acc }; }
  console.log(`  t=${String(t).padStart(2)}  acc=${( acc*100).toFixed(1).padStart(5)}%  prec=${(prec*100).toFixed(1).padStart(5)}%  rec=${(rec*100).toFixed(1).padStart(5)}%  F1=${( f1*100).toFixed(1).padStart(5)}%`);
}

console.log(`\nBest F1 at t=${bestT}: ${(bestF1*100).toFixed(1)}%`);
console.log(`  Accuracy: ${(bestStats.acc*100).toFixed(1)}%`);
console.log(`  Precision: ${(bestStats.prec*100).toFixed(1)}%  Recall: ${(bestStats.rec*100).toFixed(1)}%`);
console.log(`  TP:${bestStats.tp} TN:${bestStats.tn} FP:${bestStats.fp} FN:${bestStats.fn}`);

// ─── Baseline reminder ────────────────────────────────────────────────────────
console.log('\n─────────────────────────────────────────');
console.log('Baseline (16-vector L1 alone, t=50):');
console.log('  Accuracy: 66.6%  Precision: 55.6%  Recall: 43.7%  F1: 48.9%');
console.log('─────────────────────────────────────────');
console.log('\nNext: if Layer 4 shows signal, test combined L1+L4 scoring.');
