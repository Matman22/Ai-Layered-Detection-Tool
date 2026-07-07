/**
 * Trains the Top-1 stylometric classifier (logistic regression) on the 17
 * Layer-1 signals extracted by extract_features.js, and measures it HONESTLY
 * against the current hand-tuned weighted-sum baseline on a held-out test set.
 *
 * Pure Node — no Python/sklearn dependency. Deterministic (fixed seed).
 *
 * Reads:  eval/features.csv
 * Writes: eval/model.json   { feature_names, mean[], std[], weights[], bias }
 * Usage:  node eval/train_classifier.js
 */

const fs = require('fs');
const path = require('path');

// ─── Current hand-tuned Layer-1 weights (must match node_runner.js / index.html) ──
// 18 entries: 17 signals + a "mean slot" (mean of the other 17).
const HAND_WEIGHTS = [
  0.00, 0.11, 0.13, 0.07, 0.00, 0.00, 0.00, 0.02, 0.07, 0.12,
  0.05, 0.12, 0.04, 0.08, 0.04, 0.07, 0.03, 0.05,
];

// ─── Load features csv ───────────────────────────────────────────────────────
// Input/output configurable: node train_classifier.js [features.csv] [model.json]
const inFile = process.argv[2] || 'combined_features.csv';
const outFile = process.argv[3] || 'model_multi.json';

const csv = fs.readFileSync(path.join(__dirname, inFile), 'utf8').replace(/^﻿/, '');
const lines = csv.split('\n').filter(l => l.trim());
const header = lines[0].split(',');
const FEATURE_NAMES = header.slice(0, 17);
const li = header.indexOf('is_ai_generated');
const l2i = header.indexOf('l2_score');
const l5i = header.indexOf('l5_score');
const si = header.indexOf('source'); // -1 for single-source legacy features.csv

const data = lines.slice(1).map(line => {
  const c = line.split(',');
  return {
    x: c.slice(0, 17).map(Number),
    y: Number(c[li]), l2: Number(c[l2i]), l5: Number(c[l5i]),
    source: si !== -1 ? c[si] : 'all',
  };
});

// ─── Deterministic stratified 70/30 split ────────────────────────────────────
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function stratifiedSplit(rows, testFrac, seed) {
  const rnd = mulberry32(seed);
  const byClass = { 0: [], 1: [] };
  rows.forEach(r => byClass[r.y].push(r));
  const train = [], test = [];
  for (const k of [0, 1]) {
    const arr = byClass[k].slice();
    for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
    const nTest = Math.round(arr.length * testFrac);
    test.push(...arr.slice(0, nTest));
    train.push(...arr.slice(nTest));
  }
  return { train, test };
}

const { train, test } = stratifiedSplit(data, 0.30, 42);

// ─── Logistic regression via gradient descent (L2-regularized) ───────────────
const D = 17;
const sigmoid = t => 1 / (1 + Math.exp(-t));
const LR = 0.1, EPOCHS = 3000, LAMBDA = 0.01;

// Fit standardizer + weights on a set of rows. Returns a self-contained model.
function trainLogistic(rows) {
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
function predictP(model, r) {
  return sigmoid(r.x.reduce((s, v, j) => s + v * model.w[j] / model.std[j], 0)
    - model.w.reduce((s, wj, j) => s + wj * model.mean[j] / model.std[j], 0) + model.b);
}

// Model used for the single-split report below.
const splitModel = trainLogistic(train);
const { mean, std, w, b } = splitModel;
const z = r => r.x.map((v, j) => (v - mean[j]) / std[j]);

// ─── Scoring functions ───────────────────────────────────────────────────────
// Baseline Layer-1 composite (mirrors node_runner.js exactly).
function handL1(x) {
  const s = x.slice();
  s[17] = s.slice(0, 17).reduce((a, v) => a + v, 0) / 17; // mean slot
  return Math.min(100, s.reduce((sum, v, i) => sum + v * HAND_WEIGHTS[i], 0));
}
// Classifier Layer-1 = P(AI) * 100.
function clfL1(r) {
  return sigmoid(z(r).reduce((s, v, j) => s + v * w[j], 0) + b) * 100;
}
// Full product verdict blend + binary prediction (Mixed counts as AI, threshold T).
function verdict(l1, l2, l5, T) {
  const combined = Math.min(100, l1 * 0.75 + l2 * 0.15 + l5 * 0.10);
  return combined >= T ? 1 : 0;
}

// ─── Metrics ─────────────────────────────────────────────────────────────────
function metrics(rows, predFn) {
  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const r of rows) {
    const p = predFn(r), y = r.y;
    if (p === 1 && y === 1) tp++; else if (p === 1 && y === 0) fp++;
    else if (p === 0 && y === 0) tn++; else fn++;
  }
  const prec = tp + fp ? tp / (tp + fp) : 0;
  const rec = tp + fn ? tp / (tp + fn) : 0;
  const spec = tn + fp ? tn / (tn + fp) : 0;
  const f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
  const acc = (tp + tn) / rows.length;
  return { acc, prec, rec, spec, f1, tp, fp, tn, fn };
}
// Pick the decision threshold that maximizes F1 on the TRAIN set.
function bestThreshold(rows, l1Fn) {
  let best = { T: 50, f1: -1 };
  for (let T = 20; T <= 80; T++) {
    const m = metrics(rows, r => verdict(l1Fn(r), r.l2, r.l5, T));
    if (m.f1 > best.f1) best = { T, f1: m.f1 };
  }
  return best.T;
}

const handL1Fn = r => handL1(r.x);
const clfL1Fn = r => clfL1(r);

// Baseline at its NATIVE product threshold (40 = Mixed-or-higher → AI), plus F1-tuned.
const handT = bestThreshold(train, handL1Fn);
const clfT = bestThreshold(train, clfL1Fn);

const pct = x => (x * 100).toFixed(1) + '%';
function report(name, rows, l1Fn, T) {
  const m = metrics(rows, r => verdict(l1Fn(r), r.l2, r.l5, T));
  console.log(`\n${name}  (threshold=${T})`);
  console.log(`  Acc ${pct(m.acc)} | Prec ${pct(m.prec)} | Recall ${pct(m.rec)} | Spec ${pct(m.spec)} | F1 ${m.f1.toFixed(4)}`);
  console.log(`  TP=${m.tp} FP=${m.fp} TN=${m.tn} FN=${m.fn}`);
  return m;
}

console.log('='.repeat(64));
console.log(`Train ${train.length}  Test ${test.length}  (AI in test: ${test.filter(r => r.y === 1).length})`);
console.log('='.repeat(64));
console.log('\n--- HELD-OUT TEST SET ---');
const bHist = report('BASELINE hand-weights @ native', test, handL1Fn, 40);
const bTuned = report('BASELINE hand-weights @ F1-tuned', test, handL1Fn, handT);
const cTuned = report('CLASSIFIER (Top-1) @ F1-tuned', test, clfL1Fn, clfT);

console.log('\n' + '='.repeat(64));
const winF1 = cTuned.f1 - bTuned.f1;
console.log(`VERDICT: classifier F1 ${cTuned.f1.toFixed(4)} vs baseline ${bTuned.f1.toFixed(4)}  (${winF1 >= 0 ? '+' : ''}${(winF1).toFixed(4)})`);
console.log(`         recall ${pct(cTuned.rec)} vs ${pct(bTuned.rec)} | specificity ${pct(cTuned.spec)} vs ${pct(bTuned.spec)}`);
console.log('='.repeat(64));

// ─── 5-fold cross-validation (is the win real, or a lucky split?) ────────────
function kfold(rows, k, seed) {
  const rnd = mulberry32(seed);
  const byClass = { 0: [], 1: [] };
  rows.forEach(r => byClass[r.y].push(r));
  for (const c of [0, 1]) { const a = byClass[c]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } }
  const folds = Array.from({ length: k }, () => []);
  for (const c of [0, 1]) byClass[c].forEach((r, i) => folds[i % k].push(r)); // stratified round-robin
  return folds;
}
const folds = kfold(data, 5, 7);
const cvHand = [], cvClf = [];
for (let i = 0; i < folds.length; i++) {
  const valid = folds[i];
  const trn = folds.filter((_, j) => j !== i).flat();
  const m = trainLogistic(trn);
  const cl1 = r => predictP(m, r) * 100;
  const hT = bestThreshold(trn, r => handL1(r.x));
  const cT = bestThreshold(trn, cl1);
  cvHand.push(metrics(valid, r => verdict(handL1(r.x), r.l2, r.l5, hT)).f1);
  cvClf.push(metrics(valid, r => verdict(cl1(r), r.l2, r.l5, cT)).f1);
}
const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
const sd = a => { const m = avg(a); return Math.sqrt(avg(a.map(v => (v - m) ** 2))); };
console.log('\n--- 5-FOLD CROSS-VALIDATION (F1) ---');
console.log(`  baseline   ${avg(cvHand).toFixed(4)} ± ${sd(cvHand).toFixed(4)}`);
console.log(`  classifier ${avg(cvClf).toFixed(4)} ± ${sd(cvClf).toFixed(4)}`);
console.log(`  mean delta ${(avg(cvClf) - avg(cvHand) >= 0 ? '+' : '')}${(avg(cvClf) - avg(cvHand)).toFixed(4)}`);

// ─── Leave-one-source-out CV (the REAL cross-dataset generalization number) ──
// Train on 3 sources, test on the held-out 4th. This is the honest answer to
// "will this work on a dataset it has never seen?" — 5-fold above can't tell us
// that because every fold contains rows from all sources.
const sources = [...new Set(data.map(r => r.source))];
if (sources.length > 1) {
  console.log('\n--- LEAVE-ONE-SOURCE-OUT CV (F1, classifier vs baseline) ---');
  const losoClf = [], losoHand = [];
  for (const heldOut of sources) {
    const trn = data.filter(r => r.source !== heldOut);
    const tst = data.filter(r => r.source === heldOut);
    const m = trainLogistic(trn);
    const cl1 = r => predictP(m, r) * 100;
    const cT = bestThreshold(trn, cl1);
    const hT = bestThreshold(trn, r => handL1(r.x));
    const mc = metrics(tst, r => verdict(cl1(r), r.l2, r.l5, cT));
    const mh = metrics(tst, r => verdict(handL1(r.x), r.l2, r.l5, hT));
    losoClf.push(mc.f1); losoHand.push(mh.f1);
    console.log(`  held-out ${heldOut.padEnd(20)} clf F1 ${mc.f1.toFixed(4)} (rec ${pct(mc.rec)}, spec ${pct(mc.spec)})  vs hand ${mh.f1.toFixed(4)}`);
  }
  console.log(`  MEAN  classifier ${avg(losoClf).toFixed(4)} ± ${sd(losoClf).toFixed(4)}  |  baseline ${avg(losoHand).toFixed(4)} ± ${sd(losoHand).toFixed(4)}`);
  console.log(`  → cross-dataset generalization gap vs in-distribution 5-fold: ${(avg(cvClf) - avg(losoClf)).toFixed(4)}`);
}

// ─── Export final model — trained on ALL data (CV already gave the estimate) ──
const finalModel = trainLogistic(data);
const model = {
  note: `Top-1 stylometric logistic-regression classifier. classifierL1 = sigmoid(w·((features-mean)/std))*100, replaces the hand-weighted Layer-1 composite. Trained on ALL of eval/${inFile} (sources: ${sources.join(', ')}). 5-fold = in-distribution; LOSO = honest cross-dataset generalization.`,
  sources,
  feature_names: FEATURE_NAMES,
  mean: finalModel.mean,
  std: finalModel.std,
  weights: finalModel.w,
  bias: finalModel.b,
};
fs.writeFileSync(path.join(__dirname, outFile), JSON.stringify(model, null, 2));
console.log(`\nWrote eval/${outFile} (trained on all ${data.length} rows)`);

// Feature importance (|standardized coef|) for interpretability.
console.log('\nTop signals by |coefficient| (final model):');
FEATURE_NAMES.map((n, j) => [n, finalModel.w[j]])
  .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  .slice(0, 8)
  .forEach(([n, c]) => console.log(`  ${n.padEnd(16)} ${c >= 0 ? '+' : ''}${c.toFixed(3)}`));
