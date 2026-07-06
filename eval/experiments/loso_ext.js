/**
 * Generalized leave-one-source-out AUROC for arbitrary feature subsets.
 *
 * Reads eval/experiments/features_ext.csv (all columns before is_ai_generated
 * are features), then runs LOSO logistic regression three ways:
 *   (a) base 17 only (cols 0-16)   — must reproduce ~0.524 mean
 *   (b) all 21 features
 *   (c) 4 readability candidates alone (cols 17-20)
 *
 * Logistic-regression config identical to eval/loso_auroc.js:
 * LR=0.1, EPOCHS=3000, LAMBDA=0.01, per-feature standardization,
 * rank-sum AUROC with average-rank tie handling.
 *
 * Usage: node eval/experiments/loso_ext.js
 */

const fs = require('fs');
const path = require('path');

const csv = fs.readFileSync(path.join(__dirname, 'features_ext.csv'), 'utf8').replace(/^﻿/, '');
const lines = csv.split('\n').filter(l => l.trim());
const header = lines[0].split(',');
const li = header.indexOf('is_ai_generated');
const si = header.indexOf('source');
const D_ALL = li; // all columns before is_ai_generated are features
const allData = lines.slice(1).map(line => {
  const c = line.split(',');
  return { x: c.slice(0, D_ALL).map(Number), y: Number(c[li]), source: c[si] };
});

const sigmoid = t => 1 / (1 + Math.exp(-t));
const LR = 0.1, EPOCHS = 3000, LAMBDA = 0.01;

function trainLogistic(rows, D) {
  const mean = new Array(D).fill(0), std = new Array(D).fill(0);
  for (const r of rows) for (let j = 0; j < D; j++) mean[j] += r.x[j];
  for (let j = 0; j < D; j++) mean[j] /= rows.length;
  for (const r of rows) for (let j = 0; j < D; j++) std[j] += (r.x[j] - mean[j]) ** 2;
  for (let j = 0; j < D; j++) std[j] = Math.sqrt(std[j] / rows.length) || 1;
  const zr = r => r.x.map((v, j) => (v - mean[j]) / std[j]);
  let w = new Array(D).fill(0), b = 0;
  const X = rows.map(zr), y = rows.map(r => r.y);
  for (let e = 0; e < EPOCHS; e++) {
    const gw = new Array(D).fill(0); let gb = 0;
    for (let i = 0; i < X.length; i++) {
      const p = sigmoid(X[i].reduce((s, v, j) => s + v * w[j], 0) + b);
      const err = p - y[i];
      for (let j = 0; j < D; j++) gw[j] += err * X[i][j];
      gb += err;
    }
    for (let j = 0; j < D; j++) w[j] -= LR * (gw[j] / X.length + LAMBDA * w[j]);
    b -= LR * (gb / X.length);
  }
  return { mean, std, w, b };
}
function predictP(m, r) {
  return sigmoid(r.x.reduce((s, v, j) => s + v * m.w[j] / m.std[j], 0)
    - m.w.reduce((s, wj, j) => s + wj * m.mean[j] / m.std[j], 0) + m.b);
}

// AUROC via rank-sum (Mann-Whitney U), average ranks for ties
function auroc(scored) {
  const pos = scored.filter(s => s.y === 1).map(s => s.p);
  const neg = scored.filter(s => s.y === 0).map(s => s.p);
  if (!pos.length || !neg.length) return NaN;
  const all = scored.map((s, i) => ({ p: s.p, y: s.y, i })).sort((a, b) => a.p - b.p);
  let i = 0; const ranks = new Array(all.length);
  while (i < all.length) {
    let j = i; while (j + 1 < all.length && all[j + 1].p === all[i].p) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[all[k].i] = r;
    i = j + 1;
  }
  let sumPosRanks = 0;
  scored.forEach((s, idx) => { if (s.y === 1) sumPosRanks += ranks[idx]; });
  const nPos = pos.length, nNeg = neg.length;
  const U = sumPosRanks - nPos * (nPos + 1) / 2;
  return U / (nPos * nNeg);
}

const sources = [...new Set(allData.map(r => r.source))];

function runLOSO(label, colIdxs) {
  const D = colIdxs.length;
  const data = allData.map(r => ({ x: colIdxs.map(j => r.x[j]), y: r.y, source: r.source }));
  console.log(`\n=== ${label} (D=${D}) ===`);
  const aucs = [];
  for (const heldOut of sources) {
    const trn = data.filter(r => r.source !== heldOut);
    const tst = data.filter(r => r.source === heldOut);
    const m = trainLogistic(trn, D);
    const scored = tst.map(r => ({ p: predictP(m, r), y: r.y }));
    const a = auroc(scored);
    aucs.push(a);
    console.log(`  held-out ${heldOut.padEnd(20)} AUROC ${a.toFixed(4)}  (n=${tst.length})`);
  }
  const mean = aucs.reduce((s, v) => s + v, 0) / aucs.length;
  console.log(`  MEAN AUROC ${mean.toFixed(4)}`);
  return { aucs, mean };
}

const range = (a, b) => Array.from({ length: b - a }, (_, i) => a + i);

console.log('LEAVE-ONE-SOURCE-OUT AUROC — iteration 1 (+4 readability features)');
console.log(`Features in CSV: ${D_ALL} (${header.slice(0, D_ALL).join(', ')})`);

runLOSO('(a) base 17 only', range(0, 17));
runLOSO('(b) all 21 features', range(0, D_ALL));
runLOSO('(c) 4 readability candidates alone', range(17, D_ALL));
