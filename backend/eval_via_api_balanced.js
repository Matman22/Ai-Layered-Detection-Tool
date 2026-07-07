/**
 * Balanced gatekeeper eval for the deployed Fast-DetectGPT Space.
 *
 * Same as eval_via_api.js but samples up to PER_CLASS human AND PER_CLASS AI
 * rows per source, so every source has both classes and AUROC is defined.
 * (combined_dataset.csv is grouped by label within source, so a naive
 * head-N limit picks a single class and AUROC comes back NaN.)
 *
 * Usage:
 *   node backend/eval_via_api_balanced.js https://<user>-<space>.hf.space [--per 40]
 */

const fs = require('fs');
const path = require('path');

const BASE = process.argv[2];
if (!BASE) { console.error('Usage: node eval_via_api_balanced.js <space-url> [--per N]'); process.exit(1); }
const perArg = process.argv.indexOf('--per');
const PER_CLASS = perArg !== -1 ? parseInt(process.argv[perArg + 1]) : 40;

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
    signal: AbortSignal.timeout(30000),
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

  // Build a balanced work list: PER_CLASS of each label per source.
  const picked = {};   // src -> {0: count, 1: count}
  const work = [];
  for (let r = 1; r < records.length; r++) {
    const src = si !== -1 ? records[r][si] : 'all';
    const text = (records[r][ti] || '').trim();
    const y = Number(records[r][yi]);
    if (!text) continue;
    picked[src] = picked[src] || { 0: 0, 1: 0 };
    if (picked[src][y] >= PER_CLASS) continue;
    picked[src][y]++;
    work.push({ src, text, y });
  }

  const bySource = {}; const all = [];
  let done = 0;
  for (const item of work) {
    try {
      const out = await scoreText(item.text);
      if (!out.ok) continue;
      (bySource[item.src] = bySource[item.src] || []).push({ s: out.discrepancy, y: item.y });
      all.push({ s: out.discrepancy, y: item.y });
    } catch (e) {
      process.stderr.write(`skip (${item.src}): ${e.message}\n`);
    }
    if (++done % 25 === 0) process.stderr.write(`  scored ${done}/${work.length}\r`);
  }

  console.log('\nFAST-DETECTGPT AUROC (zero-shot, balanced sample, via Space API)\n');
  const aucs = [];
  for (const src of Object.keys(bySource).sort()) {
    const a = auroc(bySource[src]); aucs.push(a);
    const n1 = bySource[src].filter(p => p.y === 1).length;
    const n0 = bySource[src].length - n1;
    console.log(`  ${src.padEnd(20)} AUROC ${a.toFixed(4)}  (n=${bySource[src].length}, ${n0}H/${n1}AI)`);
  }
  const valid = aucs.filter(a => !Number.isNaN(a));
  const mean = valid.reduce((s, v) => s + v, 0) / valid.length;
  console.log(`\n  MEAN per-source AUROC ${mean.toFixed(4)}`);
  console.log(`  POOLED AUROC          ${auroc(all).toFixed(4)}`);
  console.log('\n  Compare: stylometry mean LOSO AUROC = 0.524 (chance).');
}

main().catch(e => { console.error(e); process.exit(1); });
