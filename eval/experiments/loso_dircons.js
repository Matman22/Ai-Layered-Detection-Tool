/**
 * Iteration 3 — direction-consistency feature selection.
 *
 * Hypothesis: features whose label direction (sign of single-feature AUROC - 0.5)
 * is inconsistent across the 3 training sources are the ones that invert on
 * unseen domains. Per LOSO fold:
 *   1. Compute single-feature AUROC of each feature within each training source.
 *   2. Rule A (strict): keep only if all 3 directions agree AND every |AUROC-0.5| >= 0.03.
 *   3. Rule B (majority): keep if >=2 of 3 agree with |AUROC-0.5| >= 0.03.
 *   4. Train logistic regression on selected features, score held-out source.
 * Also runs (a) all-21 sanity (must reproduce mean 0.5563).
 *
 * LR config identical to loso_ext.js: LR=0.1, EPOCHS=3000, LAMBDA=0.01,
 * standardize-on-train, rank-sum AUROC with average-rank ties.
 *
 * Usage: node eval/experiments/loso_dircons.js
 */

const fs = require('fs');
const path = require('path');

const csv = fs.readFileSync(path.join(__dirname, 'features_ext.csv'), 'utf8').replace(/^﻿/, '');
const lines = csv.split('\n').filter(l => l.trim());
const header = lines[0].split(',');
const li = header.indexOf('is_ai_generated');
const si = header.indexOf('source');
const D_ALL = li;
const featNames = header.slice(0, D_ALL);
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
const MARGIN = 0.03;

// Per-source single-feature AUROCs (feature value used directly as score).
const perSourceFeatAuc = {};
for (const s of sources) {
  const rows = allData.filter(r => r.source === s);
  perSourceFeatAuc[s] = featNames.map((_, j) =>
    auroc(rows.map(r => ({ p: r.x[j], y: r.y }))));
}

function selectFeatures(trainSources, rule) {
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const devs = trainSources.map(s => perSourceFeatAuc[s][j] - 0.5);
    const strong = devs.filter(d => Math.abs(d) >= MARGIN);
    if (rule === 'strict') {
      if (strong.length === 3 && (devs.every(d => d > 0) || devs.every(d => d < 0))) keep.push(j);
    } else { // majority
      const pos = strong.filter(d => d > 0).length;
      const neg = strong.filter(d => d < 0).length;
      if (pos >= 2 || neg >= 2) keep.push(j);
    }
  }
  return keep;
}

function foldAuc(heldOut, colIdxs) {
  if (!colIdxs.length) return NaN;
  const D = colIdxs.length;
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const trn = allData.filter(r => r.source !== heldOut).map(proj);
  const tst = allData.filter(r => r.source === heldOut).map(proj);
  const m = trainLogistic(trn, D);
  return auroc(tst.map(r => ({ p: predictP(m, r), y: r.y })));
}

console.log('LEAVE-ONE-SOURCE-OUT AUROC — iteration 3 (direction-consistency selection)');
console.log(`Margin |AUROC-0.5| >= ${MARGIN}\n`);

const results = { a: [], A: [], B: [] };
const selections = {};
for (const heldOut of sources) {
  const trainSrc = sources.filter(s => s !== heldOut);
  const selA = selectFeatures(trainSrc, 'strict');
  const selB = selectFeatures(trainSrc, 'majority');
  selections[heldOut] = { A: selA.map(j => featNames[j]), B: selB.map(j => featNames[j]) };
  const aAll = foldAuc(heldOut, featNames.map((_, j) => j));
  const aA = foldAuc(heldOut, selA);
  const aB = foldAuc(heldOut, selB);
  results.a.push(aAll); results.A.push(aA); results.B.push(aB);
  const n = allData.filter(r => r.source === heldOut).length;
  console.log(`held-out ${heldOut} (n=${n})`);
  console.log(`  (a) all-21          AUROC ${aAll.toFixed(4)}`);
  console.log(`  (A) strict   D=${String(selA.length).padStart(2)}   AUROC ${isNaN(aA) ? 'N/A (0 features)' : aA.toFixed(4)}`);
  console.log(`      selected: ${selections[heldOut].A.join(', ') || '(none)'}`);
  console.log(`  (B) majority D=${String(selB.length).padStart(2)}   AUROC ${isNaN(aB) ? 'N/A (0 features)' : aB.toFixed(4)}`);
  console.log(`      selected: ${selections[heldOut].B.join(', ') || '(none)'}`);
  console.log('');
}

const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
console.log('=== SUMMARY ===');
console.log('| Held-out source | (a) all-21 | (A) strict | (B) majority |');
console.log('|---|---:|---:|---:|');
sources.forEach((s, i) => {
  console.log(`| ${s} | ${results.a[i].toFixed(4)} | ${results.A[i].toFixed(4)} | ${results.B[i].toFixed(4)} |`);
});
console.log(`| **MEAN** | **${mean(results.a).toFixed(4)}** | **${mean(results.A).toFixed(4)}** | **${mean(results.B).toFixed(4)}** |`);

// Per-source single-feature AUROC diagnostic
console.log('\n=== Per-source single-feature AUROC (diagnostic) ===');
console.log(['feature', ...sources].join('\t'));
featNames.forEach((f, j) => {
  console.log([f, ...sources.map(s => perSourceFeatAuc[s][j].toFixed(3))].join('\t'));
});
