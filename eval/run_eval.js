/**
 * Evaluation pipeline for AI Origin Detector
 * Reads benchmark CSV, runs detection on each row, outputs results.csv
 * Usage: node eval/run_eval.js [--sample N]
 */

const fs = require('fs');
const path = require('path');

// ─── Parse CLI args ───────────────────────────────────────────────────────────
const sampleArg = process.argv.indexOf('--sample');
const SAMPLE_SIZE = sampleArg !== -1 ? parseInt(process.argv[sampleArg + 1]) : null;

const CSV_PATH = 'C:\\Users\\matan\\Downloads\\archive_dataset\\benchmark_ai_detection_multimodel_2026.csv';
const OUT_PATH = path.join(__dirname, 'results.csv');

// ─── CSV parser (handles quoted fields with embedded commas) ──────────────────
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
      result.push(current);
      current = '';
    } else {
      current += ch;
    }
  }
  result.push(current);
  return result;
}

// ─── Detection logic extracted from index.html ───────────────────────────────

const AI_PHRASES = [
  'it is worth noting','it is important to note','it is essential to','it should be noted',
  'in conclusion','to summarize','in summary','as mentioned earlier','as previously stated',
  'it is crucial to','one must consider','it is imperative','needless to say',
  'it goes without saying','at the end of the day','in terms of','with regard to',
  'it is evident that','it can be seen that','it is clear that','this is not to say',
  'on the other hand','furthermore','moreover','additionally','consequently',
  'therefore','thus','hence','in addition','as a result','in other words',
  'that being said','having said that','all things considered','to put it simply',
  'in light of','taking into account','it is worth mentioning','last but not least',
  'first and foremost','in the context of','with this in mind'
];

function calcPerplexity(text) {
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  if (words.length < 10) return 0;
  const freq = {};
  words.forEach(w => { freq[w] = (freq[w] || 0) + 1; });
  const uniqueRatio = Object.keys(freq).length / words.length;
  const topWordFreq = Math.max(...Object.values(freq)) / words.length;
  let score = 0;
  if (uniqueRatio < 0.4) score += 40;
  else if (uniqueRatio < 0.6) score += 20;
  if (topWordFreq > 0.05) score += 30;
  else if (topWordFreq > 0.03) score += 15;
  return Math.min(100, score);
}

function calcBurstiness(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  if (sentences.length < 4) return 50;
  const lengths = sentences.map(s => s.trim().split(/\s+/).length);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const variance = lengths.reduce((s, l) => s + Math.pow(l - mean, 2), 0) / lengths.length;
  const stdDev = Math.sqrt(variance);
  const cv = stdDev / mean;
  if (cv < 0.15) return 90;
  if (cv < 0.25) return 70;
  if (cv < 0.35) return 50;
  if (cv < 0.50) return 30;
  return 10;
}

function calcLexicalDiversity(text) {
  const words = text.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
  if (words.length < 20) return 50;
  const unique = new Set(words).size;
  const ttr = unique / words.length;
  if (ttr < 0.4) return 80;
  if (ttr < 0.55) return 60;
  if (ttr < 0.7) return 40;
  return 20;
}

function calcAIPhrases(text) {
  const lower = text.toLowerCase();
  let count = 0;
  AI_PHRASES.forEach(p => { if (lower.includes(p)) count++; });
  const density = count / (text.split(/\s+/).length / 100);
  if (density > 3) return 90;
  if (density > 1.5) return 70;
  if (density > 0.5) return 50;
  if (count > 0) return 30;
  return 5;
}

function calcHedging(text) {
  const hedges = ['may','might','could','possibly','perhaps','seemingly','apparently',
    'in some cases','it appears','suggests that','tends to','generally','typically',
    'often','usually','sometimes','arguably'];
  const lower = text.toLowerCase();
  const words = lower.match(/\b\w+\b/g) || [];
  let count = 0;
  hedges.forEach(h => { const re = new RegExp('\\b' + h + '\\b', 'g'); const m = lower.match(re); if (m) count += m.length; });
  const density = (count / words.length) * 100;
  if (density > 3) return 80;
  if (density > 1.5) return 60;
  if (density > 0.5) return 40;
  return 20;
}

function calcPassiveVoice(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  if (!sentences.length) return 50;
  const passiveRe = /\b(is|are|was|were|be|been|being)\s+\w+ed\b/gi;
  let passiveCount = 0;
  sentences.forEach(s => { const m = s.match(passiveRe); if (m) passiveCount++; });
  const ratio = passiveCount / sentences.length;
  if (ratio > 0.4) return 80;
  if (ratio > 0.25) return 60;
  if (ratio > 0.1) return 40;
  return 20;
}

function calcTransitions(text) {
  const transitions = ['however','nevertheless','furthermore','moreover','additionally',
    'consequently','therefore','thus','meanwhile','subsequently','conversely',
    'alternatively','specifically','notably','importantly'];
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  if (sentences.length < 3) return 50;
  let transCount = 0;
  const lower = text.toLowerCase();
  transitions.forEach(t => { const re = new RegExp('\\b' + t + '\\b', 'g'); const m = lower.match(re); if (m) transCount += m.length; });
  const density = (transCount / sentences.length) * 100;
  const uniformRe = /^(however|furthermore|moreover|additionally|consequently|therefore|thus)\b/i;
  let uniformStart = 0;
  sentences.forEach(s => { if (uniformRe.test(s.trim())) uniformStart++; });
  const uniformRatio = uniformStart / sentences.length;
  let score = 0;
  if (density > 20) score += 40;
  else if (density > 10) score += 25;
  else if (density > 5) score += 10;
  if (uniformRatio > 0.3) score += 40;
  else if (uniformRatio > 0.15) score += 20;
  return Math.min(100, score);
}

function calcClauseDepth(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  if (!sentences.length) return 50;
  const subordinators = ['which','that','who','whom','whose','where','when','while',
    'although','though','because','since','if','unless','until','after','before'];
  let totalDepth = 0;
  sentences.forEach(s => {
    const lower = s.toLowerCase();
    let depth = 1;
    subordinators.forEach(sub => { const re = new RegExp('\\b' + sub + '\\b', 'g'); const m = lower.match(re); if (m) depth += m.length; });
    totalDepth += depth;
  });
  const avgDepth = totalDepth / sentences.length;
  if (avgDepth > 3.5) return 80;
  if (avgDepth > 2.5) return 60;
  if (avgDepth > 1.8) return 40;
  return 20;
}

function calcPunctuationVariance(text) {
  const sentences = text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 5);
  if (sentences.length < 4) return 50;
  const punctCounts = sentences.map(s => (s.match(/[,;:\-\(\)]/g) || []).length);
  const mean = punctCounts.reduce((a, b) => a + b, 0) / punctCounts.length;
  const variance = punctCounts.reduce((s, c) => s + Math.pow(c - mean, 2), 0) / punctCounts.length;
  const cv = mean > 0 ? Math.sqrt(variance) / mean : 0;
  if (cv < 0.3) return 85;
  if (cv < 0.5) return 65;
  if (cv < 0.8) return 40;
  return 15;
}

function calcParagraphUniformity(text) {
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 30);
  if (paragraphs.length < 2) return 50;
  const lengths = paragraphs.map(p => p.trim().split(/\s+/).length);
  const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
  const cv = mean > 0 ? Math.sqrt(lengths.reduce((s, l) => s + Math.pow(l - mean, 2), 0) / lengths.length) / mean : 1;
  if (cv < 0.1) return 90;
  if (cv < 0.2) return 70;
  if (cv < 0.35) return 45;
  return 20;
}

function calcRareWords(text) {
  const commonWords = new Set(['the','be','to','of','and','a','in','that','have','it',
    'for','not','on','with','he','as','you','do','at','this','but','his','by','from',
    'they','we','say','her','she','or','an','will','my','one','all','would','there',
    'their','what','so','up','out','if','about','who','get','which','go','me','when',
    'make','can','like','time','no','just','him','know','take','people','into','year',
    'your','good','some','could','them','see','other','than','then','now','look','only',
    'come','its','over','think','also','back','after','use','two','how','our','work',
    'first','well','way','even','new','want','because','any','these','give','day','most']);
  const words = text.toLowerCase().match(/\b[a-z]{5,}\b/g) || [];
  if (words.length < 20) return 50;
  const rareCount = words.filter(w => !commonWords.has(w)).length;
  const rareRatio = rareCount / words.length;
  if (rareRatio < 0.3) return 70;
  if (rareRatio < 0.5) return 50;
  if (rareRatio < 0.7) return 30;
  return 15;
}

function calcRegisterStability(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
  if (sentences.length < 3) return 50;
  const formalMarkers = /\b(therefore|consequently|furthermore|moreover|additionally|nevertheless|notwithstanding|albeit|henceforth|aforementioned|hereby|wherein|thereof|pursuant|subsequent)\b/gi;
  const informalMarkers = /\b(gonna|wanna|gotta|kinda|sorta|yeah|yep|nope|ok|okay|stuff|things|lots|tons|pretty|really|very|just|basically|actually|literally|totally|like|so)\b/gi;
  const scores = sentences.map(s => {
    const fCount = (s.match(formalMarkers) || []).length;
    const iCount = (s.match(informalMarkers) || []).length;
    if (fCount === 0 && iCount === 0) return 0.5;
    const total = fCount + iCount;
    return fCount / total;
  }).filter(s => s !== 0.5);
  if (scores.length < 2) return 50;
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const variance = scores.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / scores.length;
  const stdDev = Math.sqrt(variance);
  if (stdDev < 0.1) return 90;
  if (stdDev < 0.2) return 70;
  if (stdDev < 0.35) return 45;
  return 20;
}

function calcNgramRepetition(text) {
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length < 20) return 50;
  const trigrams = {};
  for (let i = 0; i < words.length - 2; i++) {
    const tg = words[i] + ' ' + words[i+1] + ' ' + words[i+2];
    trigrams[tg] = (trigrams[tg] || 0) + 1;
  }
  const repeated = Object.values(trigrams).filter(c => c > 1).length;
  const totalTrigrams = words.length - 2;
  const repRatio = repeated / totalTrigrams;
  if (repRatio > 0.15) return 85;
  if (repRatio > 0.08) return 65;
  if (repRatio > 0.03) return 40;
  return 15;
}

function calcOpenerDiversity(text) {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
  if (sentences.length < 4) return 50;
  const openers = sentences.map(s => s.trim().split(/\s+/)[0].toLowerCase());
  const unique = new Set(openers).size;
  const diversity = unique / openers.length;
  if (diversity < 0.4) return 80;
  if (diversity < 0.6) return 60;
  if (diversity < 0.8) return 35;
  return 15;
}

function calcPunctuationFingerprint(text) {
  const total = text.length;
  if (total < 100) return 50;
  const commas = (text.match(/,/g) || []).length / total * 1000;
  const semicolons = (text.match(/;/g) || []).length / total * 1000;
  const colons = (text.match(/:/g) || []).length / total * 1000;
  const dashes = (text.match(/[-–—]/g) || []).length / total * 1000;
  let score = 0;
  if (commas > 8 && commas < 18) score += 20;
  if (semicolons < 0.5) score += 25;
  if (colons > 1 && colons < 5) score += 20;
  if (dashes < 1) score += 20;
  if (commas > 20 || commas < 3) score += 15;
  return Math.min(100, score);
}

function calcVocabClustering(text) {
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 30);
  if (paragraphs.length < 2) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 10);
    if (sentences.length < 4) return 50;
    const mid = Math.floor(sentences.length / 2);
    const firstHalf = new Set(sentences.slice(0, mid).join(' ').toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
    const secondHalf = new Set(sentences.slice(mid).join(' ').toLowerCase().match(/\b[a-z]{4,}\b/g) || []);
    const overlap = [...firstHalf].filter(w => secondHalf.has(w)).length;
    const union = new Set([...firstHalf, ...secondHalf]).size;
    const jaccard = union > 0 ? overlap / union : 0;
    if (jaccard > 0.5) return 70;
    if (jaccard > 0.35) return 50;
    return 25;
  }
  const vocabs = paragraphs.map(p => new Set(p.toLowerCase().match(/\b[a-z]{4,}\b/g) || []));
  let totalOverlap = 0, pairs = 0;
  for (let i = 0; i < vocabs.length; i++) {
    for (let j = i + 1; j < vocabs.length; j++) {
      const overlap = [...vocabs[i]].filter(w => vocabs[j].has(w)).length;
      const union = new Set([...vocabs[i], ...vocabs[j]]).size;
      totalOverlap += union > 0 ? overlap / union : 0;
      pairs++;
    }
  }
  const avgJaccard = pairs > 0 ? totalOverlap / pairs : 0;
  if (avgJaccard > 0.5) return 80;
  if (avgJaccard > 0.35) return 60;
  if (avgJaccard > 0.2) return 40;
  return 15;
}

// ─── Main scoring function ────────────────────────────────────────────────────
function detectAI(text) {
  if (!text || text.trim().length < 20) return { score: 50, label: 'uncertain' };

  const weights = [
    0.02, // perplexity
    0.12, // burstiness
    0.08, // lexical diversity
    0.07, // AI phrases
    0.02, // hedging
    0.02, // passive voice
    0.06, // transitions
    0.04, // clause depth
    0.10, // punctuation variance
    0.10, // paragraph uniformity
    0.04, // rare words
    0.10, // register stability
    0.05, // n-gram repetition
    0.05, // sentence opener diversity
    0.07, // punctuation fingerprint
    0.06, // vocab clustering
  ];

  const scores = [
    calcPerplexity(text),
    calcBurstiness(text),
    calcLexicalDiversity(text),
    calcAIPhrases(text),
    calcHedging(text),
    calcPassiveVoice(text),
    calcTransitions(text),
    calcClauseDepth(text),
    calcPunctuationVariance(text),
    calcParagraphUniformity(text),
    calcRareWords(text),
    calcRegisterStability(text),
    calcNgramRepetition(text),
    calcOpenerDiversity(text),
    calcPunctuationFingerprint(text),
    calcVocabClustering(text),
  ];

  const wordCount = (text.match(/\b\w+\b/g) || []).length;
  const reliability = wordCount < 100 ? 0.6 : wordCount < 300 ? 0.82 : wordCount < 600 ? 0.92 : 1.0;
  const composite = scores.reduce((sum, s, i) => sum + s * weights[i], 0);
  const adjusted = composite * reliability + 50 * (1 - reliability);

  return {
    score: Math.round(Math.min(100, adjusted)),
    vectors: scores.map((s, i) => ({ name: ['perplexity','burstiness','lexical','aiPhrases','hedging','passive','transitions','clauseDepth','punctVariance','paraUniformity','rareWords','register','ngramRep','openerDiv','punctFinger','vocabCluster'][i], score: Math.round(s), weight: weights[i] }))
  };
}

// ─── Run eval ────────────────────────────────────────────────────────────────
const raw = fs.readFileSync(CSV_PATH, 'utf8');
let rows = parseCSV(raw);

if (SAMPLE_SIZE) {
  // shuffle and take N
  for (let i = rows.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [rows[i], rows[j]] = [rows[j], rows[i]];
  }
  rows = rows.slice(0, SAMPLE_SIZE);
}

console.log(`Running eval on ${rows.length} rows...`);

const THRESHOLD = 50; // score >= 50 → predicted AI
let tp = 0, tn = 0, fp = 0, fn = 0;
const resultRows = ['id,source_model,true_label,score,predicted,correct'];

rows.forEach((row, idx) => {
  const text = row['text_content'] || '';
  const trueLabel = parseInt(row['is_ai_generated']) === 1 ? 'AI' : 'Human';
  const result = detectAI(text);
  const predicted = result.score >= THRESHOLD ? 'AI' : 'Human';
  const correct = predicted === trueLabel;

  if (trueLabel === 'AI' && predicted === 'AI') tp++;
  else if (trueLabel === 'Human' && predicted === 'Human') tn++;
  else if (trueLabel === 'Human' && predicted === 'AI') fp++;
  else fn++;

  resultRows.push(`${row['id']},${row['source_model']},${trueLabel},${result.score},${predicted},${correct}`);

  if ((idx + 1) % 50 === 0) process.stdout.write(`  ${idx + 1}/${rows.length}\r`);
});

fs.writeFileSync(OUT_PATH, resultRows.join('\n'));

// ─── Metrics ─────────────────────────────────────────────────────────────────
const total = tp + tn + fp + fn;
const accuracy = ((tp + tn) / total * 100).toFixed(1);
const precision = tp + fp > 0 ? (tp / (tp + fp) * 100).toFixed(1) : 'N/A';
const recall = tp + fn > 0 ? (tp / (tp + fn) * 100).toFixed(1) : 'N/A';
const f1 = (precision !== 'N/A' && recall !== 'N/A')
  ? (2 * parseFloat(precision) * parseFloat(recall) / (parseFloat(precision) + parseFloat(recall))).toFixed(1)
  : 'N/A';
const specificity = tn + fp > 0 ? (tn / (tn + fp) * 100).toFixed(1) : 'N/A';

console.log('\n╔══════════════════════════════════════╗');
console.log('║     AI DETECTOR — EVAL RESULTS       ║');
console.log('╠══════════════════════════════════════╣');
console.log(`║  Total samples : ${String(total).padEnd(19)}║`);
console.log(`║  Threshold     : score >= ${THRESHOLD}           ║`);
console.log('╠══════════════════════════════════════╣');
console.log(`║  Accuracy      : ${String(accuracy + '%').padEnd(19)}║`);
console.log(`║  Precision     : ${String(precision + '%').padEnd(19)}║`);
console.log(`║  Recall        : ${String(recall + '%').padEnd(19)}║`);
console.log(`║  F1 Score      : ${String(f1 + '%').padEnd(19)}║`);
console.log(`║  Specificity   : ${String(specificity + '%').padEnd(19)}║`);
console.log('╠══════════════════════════════════════╣');
console.log('║  Confusion Matrix:                   ║');
console.log(`║    TP (AI→AI)   : ${String(tp).padEnd(18)}║`);
console.log(`║    TN (H→Human) : ${String(tn).padEnd(18)}║`);
console.log(`║    FP (H→AI)    : ${String(fp).padEnd(18)}║`);
console.log(`║    FN (AI→Human): ${String(fn).padEnd(18)}║`);
console.log('╚══════════════════════════════════════╝');
console.log(`\nResults saved to: ${OUT_PATH}`);
