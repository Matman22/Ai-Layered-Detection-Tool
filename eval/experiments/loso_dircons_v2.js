/**
 * Iteration 4 — handling andythetechnerd03's frozen features.
 *
 * Diagnosis (see LOG.md iteration 4): the upstream HF dataset
 * andythetechnerd03/AI-human-text is itself pre-flattened (lowercased, ALL
 * punctuation and newlines stripped) at every offset — verified via the
 * datasets-server API at offsets 0/50k/150k/300k/450k. So 9 features that
 * depend on punctuation/paragraph/case structure return their neutral value
 * (50, commaRate 0) on every andy row. Not a fetcher bug; a source property.
 *
 * Fix candidates measured here (LR config identical to loso_dircons.js:
 * LR=0.1, EPOCHS=3000, LAMBDA=0.01, standardize-on-train, rank-sum AUROC):
 *   (a) all-21 linear baseline           — sanity, must reproduce 0.5563
 *   (B) majority direction-consistency   — sanity, must reproduce 0.6235
 *   (C) majority + explicit exclusion of frozen (source,feature) pairs from
 *       that source's direction votes. Expected ≈ identical to (B): a
 *       constant feature already has AUROC exactly 0.500 → margin fails →
 *       it already abstains. Run to confirm.
 *   (D) (C) + missing-data imputation in TRAINING: when a training source has
 *       a frozen feature, its rows' values are replaced by the mean of the
 *       other training sources' rows for that feature (→ z≈0 after
 *       standardization, so those rows stop distorting the weight). Held-out
 *       rows untouched (a test-constant feature can't affect AUROC ranking).
 *
 * Usage: node eval/experiments/loso_dircons_v2.js
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

// Detect frozen (constant) features per source — data-driven, not hardcoded.
const frozen = {}; // frozen[source] = Set of feature indices
for (const s of sources) {
  const rows = allData.filter(r => r.source === s);
  frozen[s] = new Set();
  for (let j = 0; j < D_ALL; j++) {
    if (rows.every(r => r.x[j] === rows[0].x[j])) frozen[s].add(j);
  }
}

// Per-source single-feature AUROCs
const perSourceFeatAuc = {};
for (const s of sources) {
  const rows = allData.filter(r => r.source === s);
  perSourceFeatAuc[s] = featNames.map((_, j) =>
    auroc(rows.map(r => ({ p: r.x[j], y: r.y }))));
}

// excludeFrozen: drop a source's vote entirely when the feature is frozen there.
function selectFeatures(trainSources, excludeFrozen) {
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const voters = excludeFrozen ? trainSources.filter(s => !frozen[s].has(j)) : trainSources;
    const devs = voters.map(s => perSourceFeatAuc[s][j] - 0.5);
    const strong = devs.filter(d => Math.abs(d) >= MARGIN);
    const pos = strong.filter(d => d > 0).length;
    const neg = strong.filter(d => d < 0).length;
    if (pos >= 2 || neg >= 2) keep.push(j);
  }
  return keep;
}

// impute=true → training rows from a source where feature j is frozen get the
// mean of the OTHER training sources' values for j (missing-data treatment).
function foldAuc(heldOut, colIdxs, impute) {
  if (!colIdxs.length) return NaN;
  const D = colIdxs.length;
  let trnRaw = allData.filter(r => r.source !== heldOut);
  if (impute) {
    const trnSources = sources.filter(s => s !== heldOut);
    trnRaw = trnRaw.map(r => ({ ...r, x: r.x.slice() }));
    for (const s of trnSources) {
      for (const j of frozen[s]) {
        const donors = trnRaw.filter(r => r.source !== s && !frozen[r.source].has(j));
        if (!donors.length) continue;
        const mu = donors.reduce((acc, r) => acc + r.x[j], 0) / donors.length;
        for (const r of trnRaw) if (r.source === s) r.x[j] = mu;
      }
    }
  }
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const trn = trnRaw.map(proj);
  const tst = allData.filter(r => r.source === heldOut).map(proj);
  const m = trainLogistic(trn, D);
  return auroc(tst.map(r => ({ p: predictP(m, r), y: r.y })));
}

console.log('LEAVE-ONE-SOURCE-OUT AUROC — iteration 4 (frozen-feature handling)');
console.log(`Margin |AUROC-0.5| >= ${MARGIN}\n`);
console.log('Frozen (constant) features per source:');
for (const s of sources) {
  console.log(`  ${s}: ${[...frozen[s]].map(j => featNames[j]).join(', ') || '(none)'}`);
}
console.log('');

const results = { a: [], B: [], C: [], D: [] };
for (const heldOut of sources) {
  const trainSrc = sources.filter(s => s !== heldOut);
  const selB = selectFeatures(trainSrc, false);
  const selC = selectFeatures(trainSrc, true);
  const aAll = foldAuc(heldOut, featNames.map((_, j) => j), false);
  const aB = foldAuc(heldOut, selB, false);
  const aC = foldAuc(heldOut, selC, false);
  const aD = foldAuc(heldOut, selC, true);
  results.a.push(aAll); results.B.push(aB); results.C.push(aC); results.D.push(aD);
  const n = allData.filter(r => r.source === heldOut).length;
  console.log(`held-out ${heldOut} (n=${n})`);
  console.log(`  (a) all-21                      AUROC ${aAll.toFixed(4)}`);
  console.log(`  (B) majority           D=${String(selB.length).padStart(2)}   AUROC ${aB.toFixed(4)}`);
  console.log(`  (C) maj+vote-excl      D=${String(selC.length).padStart(2)}   AUROC ${aC.toFixed(4)}`);
  console.log(`      selected: ${selC.map(j => featNames[j]).join(', ')}`);
  console.log(`  (D) C + train-impute   D=${String(selC.length).padStart(2)}   AUROC ${aD.toFixed(4)}`);
  console.log('');
}

const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
console.log('=== SUMMARY ===');
console.log('| Held-out source | (a) all-21 | (B) majority | (C) +vote-excl | (D) +impute |');
console.log('|---|---:|---:|---:|---:|');
sources.forEach((s, i) => {
  console.log(`| ${s} | ${results.a[i].toFixed(4)} | ${results.B[i].toFixed(4)} | ${results.C[i].toFixed(4)} | ${results.D[i].toFixed(4)} |`);
});
console.log(`| **MEAN** | **${mean(results.a).toFixed(4)}** | **${mean(results.B).toFixed(4)}** | **${mean(results.C).toFixed(4)}** | **${mean(results.D).toFixed(4)}** |`);
