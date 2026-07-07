/**
 * Iteration 2 — quadratic (extremeness) terms.
 *
 * Hypothesis: feature *direction* is domain-specific but *extremeness* is not.
 * Adding squared standardized terms lets logistic regression learn
 * "extreme in either direction = AI".
 *
 * Configurations (LOSO, per-source + mean AUROC):
 *   (a)  21 linear features, LAMBDA=0.01  — sanity gate, must reproduce ~0.5563
 *   (b)  21 linear + 21 squared (42), LAMBDA=0.01
 *   (b2) same 42 features, LAMBDA=0.05
 *   (c)  21 squared terms alone, LAMBDA=0.01
 *
 * Leakage guard: per fold, z = (x - mean_train) / std_train is fit ONLY on the
 * 3 training sources; z and z² are then re-standardized inside trainLogistic
 * using training rows only. Held-out rows never influence any mean/std.
 *
 * Training config identical to loso_ext.js: LR=0.1, EPOCHS=3000,
 * rank-sum AUROC with average-rank tie handling.
 *
 * Usage: node eval/experiments/loso_quad.js
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
const LR = 0.1, EPOCHS = 3000;

function trainLogistic(rows, D, lambda) {
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
    for (let j = 0; j < D; j++) w[j] -= LR * (gw[j] / X.length + lambda * w[j]);
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

// mode: 'linear' -> [z], 'quad' -> [z, z^2], 'sqonly' -> [z^2]
// z is fit on TRAINING rows only, applied to both splits.
function runLOSO(label, mode, lambda) {
  console.log(`\n=== ${label} ===`);
  const aucs = [];
  for (const heldOut of sources) {
    const trnRaw = allData.filter(r => r.source !== heldOut);
    const tstRaw = allData.filter(r => r.source === heldOut);
    // fold-local standardization fit on train only
    const mean = new Array(D_ALL).fill(0), std = new Array(D_ALL).fill(0);
    for (const r of trnRaw) for (let j = 0; j < D_ALL; j++) mean[j] += r.x[j];
    for (let j = 0; j < D_ALL; j++) mean[j] /= trnRaw.length;
    for (const r of trnRaw) for (let j = 0; j < D_ALL; j++) std[j] += (r.x[j] - mean[j]) ** 2;
    for (let j = 0; j < D_ALL; j++) std[j] = Math.sqrt(std[j] / trnRaw.length) || 1;
    const xf = r => {
      const z = r.x.map((v, j) => (v - mean[j]) / std[j]);
      const x = mode === 'linear' ? z
        : mode === 'quad' ? z.concat(z.map(v => v * v))
        : z.map(v => v * v);
      return { x, y: r.y };
    };
    const trn = trnRaw.map(xf), tst = tstRaw.map(xf);
    const D = trn[0].x.length;
    // trainLogistic re-standardizes all columns (incl. z^2) on train only
    const m = trainLogistic(trn, D, lambda);
    const scored = tst.map(r => ({ p: predictP(m, r), y: r.y }));
    const a = auroc(scored);
    aucs.push(a);
    console.log(`  held-out ${heldOut.padEnd(20)} AUROC ${a.toFixed(4)}  (n=${tst.length})`);
  }
  const mean = aucs.reduce((s, v) => s + v, 0) / aucs.length;
  console.log(`  MEAN AUROC ${mean.toFixed(4)}`);
  return { aucs, mean };
}

console.log('LEAVE-ONE-SOURCE-OUT AUROC — iteration 2 (quadratic extremeness terms)');
console.log(`Base features in CSV: ${D_ALL}`);

runLOSO('(a) 21 linear, LAMBDA=0.01 (sanity gate ~0.5563)', 'linear', 0.01);
runLOSO('(b) 21 linear + 21 squared (42), LAMBDA=0.01', 'quad', 0.01);
runLOSO('(b2) 21 linear + 21 squared (42), LAMBDA=0.05', 'quad', 0.05);
runLOSO('(c) 21 squared alone, LAMBDA=0.01', 'sqonly', 0.01);
