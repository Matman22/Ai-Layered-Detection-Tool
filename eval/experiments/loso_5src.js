/**
 * Iteration 5 — 5th training source (dmitva/human_ai_generated_text).
 *
 * Same 4 held-out folds (andythetechnerd03, RAID, MAGE, HC3); dmitva is NEVER
 * held out — it only joins the training pool. Per fold the pool = other 3
 * original sources + dmitva (4 voting sources).
 *
 * Configs:
 *   (s21) 3-src all-21 sanity      — must reproduce 0.5563
 *   (sB)  3-src majority sanity    — must reproduce 0.6235 (+-0.02)
 *   (a5)  4-src pool, all 21 linear
 *   (M3)  4-src pool, majority = >=3 of 4 agree with |AUROC-0.5| >= 0.03
 *   (M2)  4-src pool, >=2 of 4 agree, 2-2 ties broken by pooled-train direction
 *
 * LR config identical to loso_dircons.js: LR=0.1, EPOCHS=3000, LAMBDA=0.01.
 * Usage: node eval/experiments/loso_5src.js
 */

const fs = require('fs');
const path = require('path');

const csv = fs.readFileSync(path.join(__dirname, 'features_ext5.csv'), 'utf8').replace(/^﻿/, '');
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

const EXTRA = 'dmitva';
const HELD_SOURCES = [...new Set(allData.map(r => r.source))].filter(s => s !== EXTRA);

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
const allSources = [...HELD_SOURCES, EXTRA];

// Per-source single-feature AUROCs.
const perSourceFeatAuc = {};
for (const s of allSources) {
  const rows = allData.filter(r => r.source === s);
  perSourceFeatAuc[s] = featNames.map((_, j) =>
    auroc(rows.map(r => ({ p: r.x[j], y: r.y }))));
}

// Majority: more than half of training sources vote same direction with margin.
function selectMajority(trainSources) {
  const need = Math.floor(trainSources.length / 2) + 1; // 2 of 3, 3 of 4
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const devs = trainSources.map(s => perSourceFeatAuc[s][j] - 0.5);
    const strong = devs.filter(d => Math.abs(d) >= MARGIN);
    const pos = strong.filter(d => d > 0).length;
    const neg = strong.filter(d => d < 0).length;
    if (pos >= need || neg >= need) keep.push(j);
  }
  return keep;
}

// Variant: >=2 of 4 agree; 2-2 ties kept, direction irrelevant to selection
// (ties broken by pooled-train AUROC direction — used only to confirm a
// non-flat pooled signal exists: require |pooledAUROC-0.5| >= MARGIN on ties).
function selectTwoOfFour(trainSources) {
  const pooledRows = allData.filter(r => trainSources.includes(r.source));
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const devs = trainSources.map(s => perSourceFeatAuc[s][j] - 0.5);
    const strong = devs.filter(d => Math.abs(d) >= MARGIN);
    const pos = strong.filter(d => d > 0).length;
    const neg = strong.filter(d => d < 0).length;
    if (pos >= 2 && neg >= 2) {
      // 2-2 tie: break by pooled direction — keep only if pooled signal clears margin
      const pooled = auroc(pooledRows.map(r => ({ p: r.x[j], y: r.y })));
      if (Math.abs(pooled - 0.5) >= MARGIN) keep.push(j);
    } else if (pos >= 2 || neg >= 2) keep.push(j);
  }
  return keep;
}

function foldAuc(heldOut, trainSources, colIdxs) {
  if (!colIdxs.length) return NaN;
  const D = colIdxs.length;
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const trn = allData.filter(r => trainSources.includes(r.source)).map(proj);
  const tst = allData.filter(r => r.source === heldOut).map(proj);
  const m = trainLogistic(trn, D);
  return auroc(tst.map(r => ({ p: predictP(m, r), y: r.y })));
}

console.log('LEAVE-ONE-SOURCE-OUT AUROC — iteration 5 (5th training source: dmitva)');
console.log(`Margin |AUROC-0.5| >= ${MARGIN}; dmitva rows n=${allData.filter(r => r.source === EXTRA).length}\n`);

const results = { s21: [], sB: [], a5: [], M3: [], M2: [] };
const allIdx = featNames.map((_, j) => j);
for (const heldOut of HELD_SOURCES) {
  const trn3 = HELD_SOURCES.filter(s => s !== heldOut);
  const trn4 = [...trn3, EXTRA];
  const selB3 = selectMajority(trn3);
  const selM3 = selectMajority(trn4);
  const selM2 = selectTwoOfFour(trn4);
  const rS21 = foldAuc(heldOut, trn3, allIdx);
  const rSB = foldAuc(heldOut, trn3, selB3);
  const rA5 = foldAuc(heldOut, trn4, allIdx);
  const rM3 = foldAuc(heldOut, trn4, selM3);
  const rM2 = foldAuc(heldOut, trn4, selM2);
  results.s21.push(rS21); results.sB.push(rSB); results.a5.push(rA5);
  results.M3.push(rM3); results.M2.push(rM2);
  const n = allData.filter(r => r.source === heldOut).length;
  console.log(`held-out ${heldOut} (n=${n})`);
  console.log(`  (s21) 3-src all-21 sanity        AUROC ${rS21.toFixed(4)}`);
  console.log(`  (sB)  3-src majority sanity D=${String(selB3.length).padStart(2)} AUROC ${rSB.toFixed(4)}`);
  console.log(`  (a5)  4-src all-21               AUROC ${rA5.toFixed(4)}`);
  console.log(`  (M3)  4-src majority(>=3) D=${String(selM3.length).padStart(2)}   AUROC ${isNaN(rM3) ? 'N/A' : rM3.toFixed(4)}`);
  console.log(`        selected: ${selM3.map(j => featNames[j]).join(', ') || '(none)'}`);
  console.log(`  (M2)  4-src 2-of-4+ties  D=${String(selM2.length).padStart(2)}   AUROC ${isNaN(rM2) ? 'N/A' : rM2.toFixed(4)}`);
  console.log(`        selected: ${selM2.map(j => featNames[j]).join(', ') || '(none)'}`);
  console.log('');
}

const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
console.log('=== SUMMARY ===');
console.log('| Held-out source | (s21) 3-src all-21 | (sB) 3-src majority | (a5) 4-src all-21 | (M3) 4-src maj>=3 | (M2) 2-of-4+ties |');
console.log('|---|---:|---:|---:|---:|---:|');
HELD_SOURCES.forEach((s, i) => {
  console.log(`| ${s} | ${results.s21[i].toFixed(4)} | ${results.sB[i].toFixed(4)} | ${results.a5[i].toFixed(4)} | ${results.M3[i].toFixed(4)} | ${results.M2[i].toFixed(4)} |`);
});
console.log(`| **MEAN** | **${mean(results.s21).toFixed(4)}** | **${mean(results.sB).toFixed(4)}** | **${mean(results.a5).toFixed(4)}** | **${mean(results.M3).toFixed(4)}** | **${mean(results.M2).toFixed(4)}** |`);

console.log('\n=== dmitva single-feature AUROCs (diagnostic) ===');
featNames.forEach((f, j) => {
  console.log(`  ${f.padEnd(18)} ${perSourceFeatAuc[EXTRA][j].toFixed(3)}`);
});
