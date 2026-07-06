/**
 * Gatekeeper eval for the deployed Fast-DetectGPT Space — no torch needed here.
 *
 * POSTs every row of eval/combined_dataset.csv to the Space's /score endpoint
 * and reports per-source AUROC, so we can compare apples-to-apples against the
 * stylometric classifier's cross-dataset result:
 *
 *     stylometry  mean LOSO AUROC = 0.524  (chance; HC3 0.26 inverted)
 *
 * Ship the backend into index.html ONLY if Fast-DetectGPT clearly beats this.
 *
 * Usage:
 *   node backend/eval_via_api.js https://<user>-<space>.hf.space [--limit 100]
 */

const fs = require('fs');
const path = require('path');

const BASE = process.argv[2];
if (!BASE) { console.error('Usage: node eval_via_api.js <space-url> [--limit N]'); process.exit(1); }
const limArg = process.argv.indexOf('--limit');
const LIMIT = limArg !== -1 ? parseInt(process.argv[limArg + 1]) : 0;

// Full-file CSV parser (handles quoted fields spanning newlines).
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

function auroc(pairs) {
  const pos = pairs.filter(p => p.y === 1), neg = pairs.filter(p => p.y === 0);
  if (!pos.length || !neg.length) return NaN;
  const ordered = pairs.map((p, i) => ({ ...p, i })).sort((a, b) => a.s - b.s);
  const ranks = new Array(ordered.length);
  let i = 0;
  while (i < ordered.length) {
    let j = i; while (j + 1 < ordered.length && ordered[j + 1].s === ordered[i].s) j++;
    const r = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[ordered[k].i] = r;
    i = j + 1;
  }
  let sumPos = 0; pairs.forEach((p, idx) => { if (p.y === 1) sumPos += ranks[idx]; });
  const u = sumPos - pos.length * (pos.length + 1) / 2;
  return u / (pos.length * neg.length);
}

async function scoreText(text) {
  const res = await fetch(`${BASE.replace(/\/$/, '')}/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function main() {
  const csvPath = path.join(__dirname, '..', 'eval', 'combined_dataset.csv');
  const records = parseCSV(fs.readFileSync(csvPath, 'utf8').replace(/^﻿/, ''));
  const header = records[0];
  const ti = header.indexOf('text_content');
  const yi = header.indexOf('is_ai_generated');
  const si = header.indexOf('source');

  const bySource = {}; const all = []; const seen = {};
  for (let r = 1; r < records.length; r++) {
    const src = si !== -1 ? records[r][si] : 'all';
    if (LIMIT && (seen[src] || 0) >= LIMIT) continue;
    const text = (records[r][ti] || '').trim();
    const y = Number(records[r][yi]);
    if (!text) continue;
    try {
      const out = await scoreText(text);
      if (!out.ok) continue;
      (bySource[src] = bySource[src] || []).push({ s: out.discrepancy, y });
      all.push({ s: out.discrepancy, y });
      seen[src] = (seen[src] || 0) + 1;
    } catch (e) {
      process.stderr.write(`row ${r}: ${e.message}\n`);
    }
    if (r % 25 === 0) process.stderr.write(`  scored ${r}/${records.length - 1}\r`);
  }

  console.log('\nFAST-DETECTGPT AUROC (zero-shot, via Space API)\n');
  const aucs = [];
  for (const src of Object.keys(bySource).sort()) {
    const a = auroc(bySource[src]); aucs.push(a);
    console.log(`  ${src.padEnd(20)} AUROC ${a.toFixed(4)}  (n=${bySource[src].length})`);
  }
  const mean = aucs.reduce((s, v) => s + v, 0) / aucs.length;
  console.log(`\n  MEAN per-source AUROC ${mean.toFixed(4)}`);
  console.log(`  POOLED AUROC          ${auroc(all).toFixed(4)}`);
  console.log('\n  Compare: stylometry mean LOSO AUROC = 0.524 (chance).');
  console.log('  Wire into index.html only if this is clearly higher.');
}

main().catch(e => { console.error(e); process.exit(1); });
