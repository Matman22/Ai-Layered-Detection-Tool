/**
 * Calibrate app.py's CENTER / SCALE from real Fast-DetectGPT discrepancies.
 *
 * The deployed service maps a raw curvature d(x) to a 0..1 AI-probability via a
 * logistic squash:   p = 1 / (1 + exp(-(d - CENTER) / SCALE)).
 * The principled way to set CENTER/SCALE is a 1-D logistic regression of
 * label ~ discrepancy over real scored data:
 *     coef, intercept  ->  CENTER = -intercept/coef,  SCALE = 1/coef.
 * The product sees one text with no source label, so we fit on the POOLED
 * (discrepancy, label) pairs — exactly the deployed operating condition.
 *
 * Scores a balanced 40H/40AI-per-source sample through the Space (cached to
 * backend/calib_pairs.json so re-runs don't re-hit the API), fits, and prints
 * the constants plus a sanity table.
 *
 * Usage:
 *   node backend/calibrate.js https://<user>-<space>.hf.space [--per 40]
 */

const fs = require('fs');
const path = require('path');

const BASE = process.argv[2];
const perArg = process.argv.indexOf('--per');
const PER_CLASS = perArg !== -1 ? parseInt(process.argv[perArg + 1]) : 40;
const CACHE = path.join(__dirname, 'calib_pairs.json');

function parseCSV(text) {
  const records = []; let field = '', record = [], q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else q = false; } else field += ch; }
    else if (ch === '"') q = true;
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

async function scoreText(text) {
  const res = await fetch(`${BASE.replace(/\/$/, '')}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
    signal: AbortSignal.timeout(30000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function collectPairs() {
  if (fs.existsSync(CACHE)) {
    const cached = JSON.parse(fs.readFileSync(CACHE, 'utf8'));
    process.stderr.write(`using cached ${cached.length} pairs (${CACHE})\n`);
    return cached;
  }
  if (!BASE) { console.error('Need <space-url> on first run (no cache yet).'); process.exit(1); }
  const csvPath = path.join(__dirname, '..', 'eval', 'combined_dataset.csv');
  const records = parseCSV(fs.readFileSync(csvPath, 'utf8').replace(/^﻿/, ''));
  const header = records[0];
  const ti = header.indexOf('text_content');
  const yi = header.indexOf('is_ai_generated');
  const si = header.indexOf('source');

  const picked = {}; const work = [];
  for (let r = 1; r < records.length; r++) {
    const src = si !== -1 ? records[r][si] : 'all';
    const text = (records[r][ti] || '').trim();
    const y = Number(records[r][yi]);
    if (!text) continue;
    picked[src] = picked[src] || { 0: 0, 1: 0 };
    if (picked[src][y] >= PER_CLASS) continue;
    picked[src][y]++;
    work.push({ text, y });
  }

  const pairs = []; let done = 0;
  for (const item of work) {
    try {
      const out = await scoreText(item.text);
      if (out.ok) pairs.push({ d: out.discrepancy, y: item.y });
    } catch (e) { process.stderr.write(`skip: ${e.message}\n`); }
    if (++done % 25 === 0) process.stderr.write(`  scored ${done}/${work.length}\r`);
  }
  fs.writeFileSync(CACHE, JSON.stringify(pairs));
  process.stderr.write(`\ncached ${pairs.length} pairs -> ${CACHE}\n`);
  return pairs;
}

// 1-D logistic regression: label ~ d.  Standardize d for stable GD, then
// unfold the fit back to raw-d space so CENTER/SCALE apply to raw discrepancy.
function fitLogistic(pairs) {
  const ds = pairs.map(p => p.d);
  const mean = ds.reduce((s, v) => s + v, 0) / ds.length;
  const std = Math.sqrt(ds.reduce((s, v) => s + (v - mean) ** 2, 0) / ds.length) || 1;
  const X = pairs.map(p => (p.d - mean) / std);
  const Y = pairs.map(p => p.y);

  let w = 0, b = 0; const lr = 0.1, epochs = 5000, n = X.length;
  for (let e = 0; e < epochs; e++) {
    let gw = 0, gb = 0;
    for (let i = 0; i < n; i++) {
      const z = w * X[i] + b;
      const pred = 1 / (1 + Math.exp(-z));
      gw += (pred - Y[i]) * X[i];
      gb += (pred - Y[i]);
    }
    w -= lr * gw / n; b -= lr * gb / n;
  }
  // z = w*(d-mean)/std + b = (w/std)*d + (b - w*mean/std)
  const coef = w / std;
  const intercept = b - w * mean / std;
  return { coef, intercept, mean, std };
}

function auroc(pairs) {
  const s = pairs.map(p => ({ s: p.d, y: p.y }));
  const pos = s.filter(p => p.y === 1), neg = s.filter(p => p.y === 0);
  const ordered = s.map((p, i) => ({ ...p, i })).sort((a, b) => a.s - b.s);
  const ranks = new Array(ordered.length); let i = 0;
  while (i < ordered.length) {
    let j = i; while (j + 1 < ordered.length && ordered[j + 1].s === ordered[i].s) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[ordered[k].i] = r;
    i = j + 1;
  }
  let sp = 0; s.forEach((p, idx) => { if (p.y === 1) sp += ranks[idx]; });
  return (sp - pos.length * (pos.length + 1) / 2) / (pos.length * neg.length);
}

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (pos - lo);
}

async function main() {
  const pairs = await collectPairs();
  const { coef, intercept } = fitLogistic(pairs);
  const CENTER = -intercept / coef;
  const SCALE = 1 / coef;

  const hum = pairs.filter(p => p.y === 0).map(p => p.d).sort((a, b) => a - b);
  const ai = pairs.filter(p => p.y === 1).map(p => p.d).sort((a, b) => a - b);
  const fpr5 = quantile(hum, 0.95);   // discrepancy threshold at 5% human FPR

  const P = d => 1 / (1 + Math.exp(-(d - CENTER) / SCALE));
  const acc = pairs.filter(p => (P(p.d) >= 0.5 ? 1 : 0) === p.y).length / pairs.length;
  const tprAt5 = ai.filter(d => d >= fpr5).length / ai.length;

  console.log('\nFAST-DETECTGPT CALIBRATION');
  console.log(`  pairs           ${pairs.length}  (${hum.length}H / ${ai.length}AI)`);
  console.log(`  pooled AUROC    ${auroc(pairs).toFixed(4)}`);
  console.log(`  human d  median ${quantile(hum, 0.5).toFixed(3)}  (p95 ${fpr5.toFixed(3)})`);
  console.log(`  AI    d  median ${quantile(ai, 0.5).toFixed(3)}`);
  console.log('\n  1-D logistic fit (label ~ discrepancy):');
  console.log(`    coef       ${coef.toFixed(4)}`);
  console.log(`    intercept  ${intercept.toFixed(4)}`);
  console.log('\n  >>> paste into backend/app.py:');
  console.log(`      CENTER = ${CENTER.toFixed(4)}`);
  console.log(`      SCALE  = ${SCALE.toFixed(4)}`);
  console.log('\n  sanity:');
  console.log(`    accuracy @ p>=0.5   ${(acc * 100).toFixed(1)}%`);
  console.log(`    TPR @ 5% human FPR  ${(tprAt5 * 100).toFixed(1)}%  (threshold d=${fpr5.toFixed(3)})`);
  console.log('    p(d) at percentiles:');
  for (const q of [0.05, 0.25, 0.5, 0.75, 0.95]) {
    const d = quantile(pairs.map(p => p.d).sort((a, b) => a - b), q);
    console.log(`      d[${(q * 100).toString().padStart(2)}%]=${d.toFixed(3)}  ->  p=${P(d).toFixed(3)}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
