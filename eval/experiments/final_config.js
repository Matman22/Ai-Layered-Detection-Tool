/**
 * Iteration 7 (FINAL) — pick the config that generalizes best to unseen sources.
 *
 * Decision metric: AUROC on source6 (artem9k/ai-text-detection-pile,
 * features_src6.csv — never used in any training or rule-tuning), training
 * ALWAYS on the 5 pooled features_ext5.csv sources.
 * Reference metric: mean 5-source LOSO AUROC on features_ext5.csv (each of the
 * 5 sources held out in turn, config's selection re-run per fold).
 *
 * Configs (pre-registered, exactly these):
 *   C1: all 21 features, no selection (iter-6 V2 reproduction, expect ~0.5317)
 *   C2: iteration-3 direction-consistency MAJORITY rule, adapted to the pooled
 *       5-source train: keep feature if a strict majority of training sources
 *       (>=3 of 5 pooled; >=3 of 4 inside LOSO folds — same convention as
 *       iter-5's M3) vote the same direction with |AUROC-0.5| >= 0.03.
 *       (With 3 training sources, iter 3's implemented rule pos>=2||neg>=2 IS
 *        the strict majority; >=2 of 5 would instead reproduce M2's looseness.)
 *   C3: iteration-5 M2 rule (>=2 same-direction strong votes; exact 2-2 tie
 *       broken by pooled-train direction) — recomputed trivially with the
 *       exact selectM2 from validate_src6.js (expect 0.4508 on source6).
 *   C4: all 21 minus features constant/degenerate on the pooled TRAINING set
 *       (std < 1e-9), no other selection.
 *
 * Math (loadCsv, trainLogistic, predictP, auroc, selectM2) copied verbatim
 * from validate_src6.js. LR=0.1, EPOCHS=3000, LAMBDA=0.01, standardize on
 * train only, rank-sum AUROC with ties.
 * Usage: node eval/experiments/final_config.js
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
  console.error('ABORT: feature schema mismatch between features_ext5.csv and features_src6.csv');
  process.exit(1);
}
const featNames = ext5.featNames;
const D_ALL = ext5.D_ALL;
const allData = ext5.rows;
const SOURCES = [...new Set(allData.map(r => r.source))]; // 5 sources
console.log(`Sources (pooled train): ${SOURCES.join(', ')}`);
console.log(`ext5 rows: ${allData.length}; src6 rows: ${src6.rows.length}\n`);

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

// Per-source single-feature AUROCs (over all 5 ext5 sources).
const perSourceFeatAuc = {};
for (const s of SOURCES) {
  const rows = allData.filter(r => r.source === s);
  perSourceFeatAuc[s] = featNames.map((_, j) =>
    auroc(rows.map(r => ({ p: r.x[j], y: r.y }))));
}

// ── Selection rules ──────────────────────────────────────────────────────────
const allIdx = featNames.map((_, j) => j);

// C2: iteration-3 majority — strict majority of training sources agree in
// direction with margin >= MARGIN. Majority = floor(n/2)+1 (2 of 3, 3 of 4, 3 of 5).
function selectMajority(trainSources) {
  const need = Math.floor(trainSources.length / 2) + 1;
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

// C3: iteration-5 M2 rule — verbatim from validate_src6.js.
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

// C4: drop features with std < 1e-9 on the given training rows.
function selectNonDegenerate(trainRows) {
  const keep = [];
  for (let j = 0; j < D_ALL; j++) {
    const vals = trainRows.map(r => r.x[j]);
    const mu = vals.reduce((s, v) => s + v, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mu) ** 2, 0) / vals.length);
    if (sd >= 1e-9) keep.push(j);
  }
  return keep;
}

// ── Scoring helpers ──────────────────────────────────────────────────────────
function fitAndScore(trainRows, testRows, colIdxs) {
  if (!colIdxs.length) return NaN;
  const proj = r => ({ x: colIdxs.map(j => r.x[j]), y: r.y });
  const m = trainLogistic(trainRows.map(proj), colIdxs.length);
  return auroc(testRows.map(proj).map(r => ({ p: predictP(m, r), y: r.y })));
}

// selector(trainSources, trainRows) -> column indices
function scoreSource6(selector) {
  const sel = selector(SOURCES, allData);
  return { auc: fitAndScore(allData, src6.rows, sel), sel };
}
function loso5(selector) {
  const per = {};
  for (const heldOut of SOURCES) {
    const trainSources = SOURCES.filter(s => s !== heldOut);
    const trainRows = allData.filter(r => trainSources.includes(r.source));
    const testRows = allData.filter(r => r.source === heldOut);
    const sel = selector(trainSources, trainRows);
    per[heldOut] = { auc: fitAndScore(trainRows, testRows, sel), D: sel.length };
  }
  const vals = Object.values(per).map(p => p.auc);
  return { mean: vals.reduce((s, v) => s + v, 0) / vals.length, per };
}

const CONFIGS = {
  C1: { name: 'all-21, no selection', selector: () => allIdx },
  C2: { name: 'iter-3 direction-consistency majority', selector: (ts) => selectMajority(ts) },
  C3: { name: 'iter-5 M2 (>=2 votes + tie-break)', selector: (ts, tr) => selectM2(ts, tr) },
  C4: { name: 'all-21 minus degenerate (std<1e-9 on train)', selector: (ts, tr) => selectNonDegenerate(tr) },
};

const results = {};
for (const [id, cfg] of Object.entries(CONFIGS)) {
  const s6 = scoreSource6(cfg.selector);
  const lo = loso5(cfg.selector);
  results[id] = { ...cfg, s6, lo };
  console.log(`${id} — ${cfg.name}`);
  console.log(`  pooled-5 selection D=${s6.sel.length}: ${s6.sel.map(j => featNames[j]).join(', ')}`);
  console.log(`  source6 AUROC = ${s6.auc.toFixed(4)}   (DECISION METRIC)`);
  console.log(`  LOSO-5 mean   = ${lo.mean.toFixed(4)}   (reference)`);
  for (const [s, p] of Object.entries(lo.per)) {
    console.log(`    held-out ${s} (D=${p.D}): ${p.auc.toFixed(4)}`);
  }
  console.log('');
}

console.log('=== FINAL RESULTS TABLE ===');
console.log('| Config | Description | source6 AUROC (decision) | LOSO-5 mean (ref) | D (pooled) |');
console.log('|---|---|---:|---:|---:|');
for (const [id, r] of Object.entries(results)) {
  console.log(`| ${id} | ${r.name} | ${r.s6.auc.toFixed(4)} | ${r.lo.mean.toFixed(4)} | ${r.s6.sel.length} |`);
}

const winner = Object.entries(results).sort((a, b) => b[1].s6.auc - a[1].s6.auc)[0];
console.log(`\nWINNER (highest source6 AUROC): ${winner[0]} — ${winner[1].name} @ ${winner[1].s6.auc.toFixed(4)}`);
if (winner[1].s6.auc <= 0.55) {
  console.log('VERDICT: best source6 AUROC <= 0.55 -> NO stylometric config generalizes. ' +
    'Stylometry ceiling confirmed on external data; LM backend (Fast-DetectGPT) is the path forward.');
} else {
  console.log('VERDICT: winner exceeds 0.55 on the external source -> ship this config as the stylometric baseline.');
}
