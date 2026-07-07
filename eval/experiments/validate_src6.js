/**
 * Iteration 6 (retry) — VALIDATION CHECKPOINT against a genuinely held-out
 * fresh source (artem9k/ai-text-detection-pile), never touched in training.
 *
 * V1: train ONCE on all 5 existing training sources (features_ext5.csv:
 *     andythetechnerd03, RAID, MAGE, HC3, dmitva) using the iteration-5 M2
 *     selection rule generalized to 5 sources (feature kept if >=2 sources
 *     give a valid single-feature-AUROC direction vote with margin >= 0.03;
 *     2-2-tie-with-1-abstain style ties broken by pooled-train direction),
 *     then score features_src6.csv. Report AUROC.
 * V2: same training pool, all-21 features, no selection. Report AUROC.
 * V3: sanity — re-run iteration-5 M2 LOSO on features_ext5.csv, confirm mean
 *     ~0.6682 +/- 0.02.
 *
 * LR config identical to loso_5src.js: LR=0.1, EPOCHS=3000, LAMBDA=0.01.
 * Usage: node eval/experiments/validate_src6.js
 */

const fs = require('fs');
const path = require('path');

function loadCsv(file) {
  const csv = fs.readFileSync(path.join(__dirname, file), 'utf8').replace(/^﻿/, '');
  const lines = csv.split('\n').filter(l => l.trim());
  const header = lines[0].split(',');
  const li = header.indexOf('is_ai_generated');
  const si = header.indexOf('source');
  const D_ALL = li;
  const featNames = header.slice(0, D_ALL);
  const rows = lines.slice(1).map(line => {
    const c = line.split(',');
    return { x: c.slice(0, D_ALL).map(Number), y: Number(c[li]), source: c[si] };
  });
  return { featNames, D_ALL, rows };
}

const ext5 = loadCsv('features_ext5.csv');
const src6 = loadCsv('features_src6.csv');
if (JSON.stringify(ext5.featNames) !== JSON.stringify(src6.featNames)) {
  console.error('ABORT: feature name/order mismatch between features_ext5.csv and features_src6.csv');
  console.error('ext5:', ext5.featNames);
  console.error('src6:', src6.featNames);
  process.exit(1);
}
const featNames = ext5.featNames;
const D_ALL = ext5.D_ALL;
const allData = ext5.rows;
const TRAIN_SOURCES = [...new Set(allData.map(r => r.source))]; // andythetechnerd03, RAID, MAGE, HC3, dmitva
console.log(`Training sources (all 5, pooled): ${TRAIN_SOURCES.join(', ')}`);
console.log(`features_ext5.csv rows: ${allData.length}; features_src6.csv rows: ${src6.rows.length}\n`);

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

const MARGIN = 0.03;

// Per-source single-feature AUROCs (over the 5 training sources).
const perSourceFeatAuc = {};
for (const s of TRAIN_SOURCES) {
  const rows = allData.filter(r => r.source === s);
  perSourceFeatAuc[s] = featNames.map((_, j) =>
    auroc(rows.map(r => ({ p: r.x[j], y: r.y }))));
}

// Generalized iteration-5 M2 rule: >=2 sources with |AUROC-0.5|>=MARGIN in the
// same direction => keep. If the strong votes split exactly 2-vs-2 (a genuine
// tie), break by pooled-train direction (kept only if pooled |AUROC-0.5| >=
// MARGIN). With 5 sources this generalizes loso_5src.js's selectTwoOfFour
// (there defined for exactly 4 training sources) to any source count: any
// non-2-2 outcome with >=2 on one side is decided outright; only an exact 2-2
// split needs the tie-break.
function selectM2(trainSources, pooledRows) {
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const devs = trainSources.map(s => perSourceFeatAuc[s][j] - 0.5);
    const strong = devs.filter(d => Math.abs(d) >= MARGIN);
    const pos = strong.filter(d => d > 0).length;
    const neg = strong.filter(d => d < 0).length;
    if (pos >= 2 && neg >= 2) {
      const pooled = auroc(pooledRows.map(r => ({ p: r.x[j], y: r.y })));
      if (Math.abs(pooled - 0.5) >= MARGIN) keep.push(j);
    } else if (pos >= 2 || neg >= 2) keep.push(j);
  }
  return keep;
}

const allIdx = featNames.map((_, j) => j);

// ─── V1: 5-source pool, M2 selection, score on src6 ─────────────────────────
const selV1 = selectM2(TRAIN_SOURCES, allData);
function scoreExternal(colIdxs, testRows) {
  const D = colIdxs.length;
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const trn = allData.map(proj);
  const tst = testRows.map(proj);
  const m = trainLogistic(trn, D);
  return auroc(tst.map(r => ({ p: predictP(m, r), y: r.y })));
}
const v1 = scoreExternal(selV1, src6.rows);
console.log(`(V1) M2-selected (D=${selV1.length}) trained on 5-source pool, scored on source6:`);
console.log(`     selected: ${selV1.map(j => featNames[j]).join(', ')}`);
console.log(`     AUROC = ${v1.toFixed(4)}\n`);

// ─── V2: all-21 features, no selection, score on src6 ───────────────────────
const v2 = scoreExternal(allIdx, src6.rows);
console.log(`(V2) all-21 trained on 5-source pool, scored on source6:`);
console.log(`     AUROC = ${v2.toFixed(4)}\n`);

// ─── V3: sanity — reproduce iteration-5 M2 LOSO on features_ext5.csv ────────
const EXTRA = 'dmitva';
const HELD_SOURCES = TRAIN_SOURCES.filter(s => s !== EXTRA);

function foldAuc(heldOut, trainSources, colIdxs) {
  if (!colIdxs.length) return NaN;
  const D = colIdxs.length;
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const trn = allData.filter(r => trainSources.includes(r.source)).map(proj);
  const tst = allData.filter(r => r.source === heldOut).map(proj);
  const m = trainLogistic(trn, D);
  return auroc(tst.map(r => ({ p: predictP(m, r), y: r.y })));
}

const v3results = [];
console.log('(V3) sanity — iteration-5 M2 LOSO reproduction on features_ext5.csv:');
for (const heldOut of HELD_SOURCES) {
  const trn4 = [...HELD_SOURCES.filter(s => s !== heldOut), EXTRA];
  const pooledRows = allData.filter(r => trn4.includes(r.source));
  const selM2 = selectM2(trn4, pooledRows);
  const rM2 = foldAuc(heldOut, trn4, selM2);
  v3results.push(rM2);
  console.log(`     held-out ${heldOut} (D=${selM2.length}): AUROC = ${rM2.toFixed(4)}`);
}
const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
const v3mean = mean(v3results);
console.log(`     MEAN = ${v3mean.toFixed(4)} (expect ~0.6682 +/- 0.02)\n`);

console.log('=== SUMMARY ===');
console.log(`V1 (M2-selected, external validation on source6): ${v1.toFixed(4)}`);
console.log(`V2 (all-21, external validation on source6):      ${v2.toFixed(4)}`);
console.log(`V3 (sanity LOSO mean, features_ext5.csv):          ${v3mean.toFixed(4)}`);

let branch;
if (v1 >= 0.60) branch = 'V1 >= 0.60 -> gains are real';
else if (v1 > 0.52) branch = 'V1 in 0.52-0.60 -> weak transfer, flag caution';
else branch = 'V1 <= 0.52 -> recent rule-tuning is meta-overfit, recommend rollback to iteration-3 (0.6235)';
console.log(`\nInterpretation branch: ${branch}`);
console.log(`V2 >= V1: ${v2 >= v1} (selection adds nothing out-of-domain if true)`);
