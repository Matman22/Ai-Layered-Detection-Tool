/**
 * Diagnostic: leave-one-source-out AUROC for the L1 classifier.
 *
 * Thresholded F1 collapsing across sources can mean two very different things:
 *   (a) CALIBRATION problem — the model RANKS AI>human fine, but the single
 *       global threshold sits in the wrong place per domain. AUROC stays high.
 *       → rescuable with per-domain calibration.
 *   (b) RANKING problem — the features genuinely don't separate AI from human
 *       on an unseen domain. AUROC ~0.5.
 *       → stylometry has hit its ceiling; need a real LM signal.
 *
 * AUROC is threshold-free, so it isolates (a) from (b).
 */

const fs = require('fs');
const path = require('path');

const csv = fs.readFileSync(path.join(__dirname, 'combined_features.csv'), 'utf8').replace(/^﻿/, '');
const lines = csv.split('\n').filter(l => l.trim());
const header = lines[0].split(',');
const li = header.indexOf('is_ai_generated');
const si = header.indexOf('source');
const data = lines.slice(1).map(line => {
  const c = line.split(',');
  return { x: c.slice(0, 17).map(Number), y: Number(c[li]), source: c[si] };
});

const D = 17;
const sigmoid = t => 1 / (1 + Math.exp(-t));
const LR = 0.1, EPOCHS = 3000, LAMBDA = 0.01;

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
function predictP(m, r) {
  return sigmoid(r.x.reduce((s, v, j) => s + v * m.w[j] / m.std[j], 0)
    - m.w.reduce((s, wj, j) => s + wj * m.mean[j] / m.std[j], 0) + m.b);
}

// AUROC via rank-sum (Mann-Whitney U)
function auroc(scored) {
  const pos = scored.filter(s => s.y === 1).map(s => s.p);
  const neg = scored.filter(s => s.y === 0).map(s => s.p);
  if (!pos.length || !neg.length) return NaN;
  const all = scored.map((s, i) => ({ p: s.p, y: s.y, i })).sort((a, b) => a.p - b.p);
  // assign ranks (average for ties)
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

const sources = [...new Set(data.map(r => r.source))];
console.log('LEAVE-ONE-SOURCE-OUT AUROC (threshold-free ranking quality)\n');
const aucs = [];
for (const heldOut of sources) {
  const trn = data.filter(r => r.source !== heldOut);
  const tst = data.filter(r => r.source === heldOut);
  const m = trainLogistic(trn);
  const scored = tst.map(r => ({ p: predictP(m, r), y: r.y }));
  const a = auroc(scored);
  aucs.push(a);
  console.log(`  held-out ${heldOut.padEnd(20)} AUROC ${a.toFixed(4)}  (n=${tst.length})`);
}
const avg = a => a.reduce((s, v) => s + v, 0) / a.length;
console.log(`\n  MEAN AUROC ${avg(aucs).toFixed(4)}`);
console.log('\n  Interpretation:');
console.log('    ~0.50 = features do not separate AI/human on unseen domain (ranking ceiling)');
console.log('    >0.75 = good ranking; collapse is a CALIBRATION/threshold problem (rescuable)');
