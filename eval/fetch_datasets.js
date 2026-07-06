/**
 * Fetches AI vs Human text from 4 HuggingFace datasets via the public
 * datasets-server REST API (no key, no Python required).
 *
 * Writes: eval/combined_dataset.csv
 *   columns: text_content, is_ai_generated, source
 *
 * Usage: node eval/fetch_datasets.js [--rows-per-source N]
 *
 * Rows per source default: 300 (150 AI + 150 human, stratified).
 * HC3 is smaller — capped at what's available.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const BATCH_DELAY_MS = 1200; // pause between HF API calls to avoid 429

const OUT_PATH = path.join(__dirname, 'combined_dataset.csv');
const BATCH_SIZE = 100;

const rowsArg = process.argv.indexOf('--rows-per-source');
const ROWS_PER_SOURCE = rowsArg !== -1 ? parseInt(process.argv[rowsArg + 1]) : 300;
const PER_CLASS = Math.floor(ROWS_PER_SOURCE / 2); // balanced AI + human per source

// ─── Dataset configs ─────────────────────────────────────────────────────────

const SOURCES = [
  {
    name: 'andythetechnerd03',
    // Read from the existing fetched CSV — avoids re-fetching and a known column-name ambiguity.
    type: 'local_csv',
    csvPath: path.join(__dirname, 'real_dataset.csv'),
    // real_dataset.csv columns: text_content, is_ai_generated, source_label
    textCol: 'text_content',
    labelCol: 'is_ai_generated',
  },
  {
    name: 'RAID',
    dataset: 'liamdugan/raid',
    config: 'raid',
    split: 'train',
    type: 'standard',
    textCol: 'generation',
    // model='human' → 0, any model name → 1
    labelFn: row => (row.model === 'human' ? 0 : 1),
    // Only clean (non-attacked) samples for training
    filterFn: row => row.attack === 'none',
  },
  {
    name: 'MAGE',
    dataset: 'yaful/MAGE',
    config: 'default',
    split: 'train',
    type: 'standard',
    textCol: 'text',
    // WARNING: row.label is INVERTED in this dataset (1 = human).
    // Derive from src field instead: ends in '_human' → 0, else → 1.
    labelFn: row => (row.src && row.src.endsWith('_human') ? 0 : 1),
    filterFn: () => true,
    // Dataset is sorted: human rows first (~160k), AI rows after.
    // Jump ahead so we don't scan 160k rows looking for AI examples.
    aiScanOffset: 160000,
  },
  {
    name: 'HC3',
    dataset: 'Hello-SimpleAI/HC3',
    config: 'all',
    split: 'train',
    // Each row has human_answers[] and chatgpt_answers[] — explode into separate rows.
    type: 'hc3',
    filterFn: () => true,
  },
];

// ─── HTTP helper ──────────────────────────────────────────────────────────────

async function fetchRows(dataset, config, split, offset, length, retries = 4) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    const result = await new Promise((resolve, reject) => {
      const url = `https://datasets-server.huggingface.co/rows` +
        `?dataset=${encodeURIComponent(dataset)}` +
        `&config=${encodeURIComponent(config)}` +
        `&split=${encodeURIComponent(split)}` +
        `&offset=${offset}&length=${length}`;
      https.get(url, { headers: { 'User-Agent': 'node.js' } }, res => {
        let data = '';
        res.on('data', c => data += c);
        res.on('end', () => resolve({ status: res.statusCode, data }));
      }).on('error', reject);
    });

    if (result.status === 429 || result.data.startsWith('<!DOCTYPE')) {
      const wait = (attempt + 1) * 8000;
      process.stdout.write(`\n  [rate-limited] waiting ${wait/1000}s before retry ${attempt+1}/${retries}...\r`);
      await sleep(wait);
      continue;
    }

    try { return JSON.parse(result.data); }
    catch (e) { throw new Error('JSON parse error: ' + result.data.slice(0, 200)); }
  }
  throw new Error('Rate limit retries exhausted');
}

// ─── CSV helper ───────────────────────────────────────────────────────────────

function escapeCSV(val) {
  if (val === null || val === undefined) return '';
  const str = String(val).replace(/"/g, '""');
  return (str.includes(',') || str.includes('"') || str.includes('\n')) ? `"${str}"` : str;
}

// ─── Fetch one standard source (balanced) ────────────────────────────────────

async function fetchClass(src, targetLabel, startOffset) {
  const { name, dataset, config, split, textCol, labelFn, filterFn } = src;
  const className = targetLabel === 0 ? 'human' : 'AI';
  const collected = [];
  let offset = startOffset;
  let totalFetched = 0;

  while (collected.length < PER_CLASS) {
    const result = await fetchRows(dataset, config, split, offset, BATCH_SIZE);
    if (result.error || !result.rows || result.rows.length === 0) break;

    for (const { row } of result.rows) {
      if (!filterFn(row)) continue;
      const text = String(row[textCol] || '').trim();
      if (text.length < 50) continue;
      const label = labelFn(row);
      if (label === targetLabel && collected.length < PER_CLASS) collected.push(text);
    }

    totalFetched += result.rows.length;
    offset += BATCH_SIZE;
    await sleep(BATCH_DELAY_MS);
    process.stdout.write(`  [${name}] ${className}=${collected.length}/${PER_CLASS}  (scanned ${totalFetched})\r`);

    if (collected.length >= PER_CLASS) break;
    if (totalFetched >= 8000) { console.log(`\n  [${name}] scan limit for ${className}`); break; }
  }
  return collected;
}

async function fetchStandard(src) {
  const { name, aiScanOffset = 0 } = src;
  console.log(`\n[${name}] Fetching up to ${PER_CLASS} AI + ${PER_CLASS} human...`);

  // Fetch both classes in parallel (different offsets) — halves wall-clock time.
  const [humanTexts, aiTexts] = await Promise.all([
    fetchClass(src, 0, 0),
    fetchClass(src, 1, aiScanOffset),
  ]);

  const rows = [];
  for (const text of humanTexts) rows.push({ text, label: 0, source: name });
  for (const text of aiTexts)    rows.push({ text, label: 1, source: name });

  console.log(`\n  [${name}] collected human=${humanTexts.length}  AI=${aiTexts.length}`);
  return rows;
}

// ─── Fetch HC3 (explode human_answers / chatgpt_answers) ─────────────────────

async function fetchHC3(src) {
  const { name, dataset, config, split } = src;
  console.log(`\n[${name}] Fetching (explode human+chatgpt answers)...`);

  const buckets = { 0: [], 1: [] };
  let offset = 0;
  let totalFetched = 0;

  while (buckets[0].length < PER_CLASS || buckets[1].length < PER_CLASS) {
    const result = await fetchRows(dataset, config, split, offset, BATCH_SIZE);
    if (result.error || !result.rows || result.rows.length === 0) break;

    for (const { row } of result.rows) {
      if (buckets[0].length < PER_CLASS) {
        for (const text of (row.human_answers || [])) {
          const t = text.trim();
          if (t.length >= 50 && buckets[0].length < PER_CLASS) buckets[0].push(t);
        }
      }
      if (buckets[1].length < PER_CLASS) {
        for (const text of (row.chatgpt_answers || [])) {
          const t = text.trim();
          if (t.length >= 50 && buckets[1].length < PER_CLASS) buckets[1].push(t);
        }
      }
    }

    totalFetched += result.rows.length;
    offset += BATCH_SIZE;
    await sleep(BATCH_DELAY_MS);
    process.stdout.write(`  human=${buckets[0].length}/${PER_CLASS}  AI=${buckets[1].length}/${PER_CLASS}  (scanned ${totalFetched} QA pairs)\r`);

    if (buckets[0].length >= PER_CLASS && buckets[1].length >= PER_CLASS) break;
    if (totalFetched >= 5000) break;
  }

  const rows = [];
  for (const text of buckets[0]) rows.push({ text, label: 0, source: name });
  for (const text of buckets[1]) rows.push({ text, label: 1, source: name });

  console.log(`\n  [${name}] collected human=${buckets[0].length}  AI=${buckets[1].length}`);
  return rows;
}

// ─── Load local CSV (andythetechnerd03 already fetched) ──────────────────────

function loadLocalCSV(src) {
  const { name, csvPath, textCol, labelCol } = src;
  console.log(`\n[${name}] Reading from ${csvPath}...`);
  const raw = fs.readFileSync(csvPath, 'utf8').replace(/^﻿/, '');
  const lines = raw.split('\n').filter(l => l.trim());
  const headers = lines[0].split(',');
  const ti = headers.indexOf(textCol);
  const li = headers.indexOf(labelCol);

  // Parse CSV respecting quoted fields
  const parseCSVLine = line => {
    const fields = [];
    let cur = '', inQ = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') { inQ = !inQ; }
      else if (ch === ',' && !inQ) { fields.push(cur); cur = ''; }
      else cur += ch;
    }
    fields.push(cur);
    return fields;
  };

  const rows = [];
  for (const line of lines.slice(1)) {
    const fields = parseCSVLine(line);
    const text = (fields[ti] || '').trim();
    const label = parseInt(fields[li]);
    if (text.length >= 50 && (label === 0 || label === 1)) {
      rows.push({ text, label, source: name });
    }
  }
  console.log(`  [${name}] loaded ${rows.length} rows (AI=${rows.filter(r=>r.label===1).length} human=${rows.filter(r=>r.label===0).length})`);
  return rows;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`fetch_datasets.js — target ${ROWS_PER_SOURCE} rows/source (${PER_CLASS} AI + ${PER_CLASS} human)`);
  console.log('='.repeat(64));

  const allRows = [];

  for (const src of SOURCES) {
    let rows;
    if (src.type === 'local_csv') {
      rows = loadLocalCSV(src);
    } else if (src.type === 'hc3') {
      rows = await fetchHC3(src);
    } else {
      rows = await fetchStandard(src);
    }
    allRows.push(...rows);
  }

  // Write CSV
  const header = 'text_content,is_ai_generated,source';
  const csvLines = [header];
  for (const { text, label, source } of allRows) {
    csvLines.push(`${escapeCSV(text)},${label},${escapeCSV(source)}`);
  }
  fs.writeFileSync(OUT_PATH, csvLines.join('\n'), 'utf8');

  // Summary
  console.log('\n' + '='.repeat(64));
  console.log(`Wrote ${allRows.length} rows to ${OUT_PATH}`);
  console.log('\nDistribution by source:');
  const bySource = {};
  for (const { label, source } of allRows) {
    if (!bySource[source]) bySource[source] = { ai: 0, human: 0 };
    if (label === 1) bySource[source].ai++;
    else bySource[source].human++;
  }
  for (const [src, counts] of Object.entries(bySource)) {
    const total = counts.ai + counts.human;
    const aiPct = ((counts.ai / total) * 100).toFixed(1);
    console.log(`  ${src.padEnd(20)} total=${total}  AI=${counts.ai} (${aiPct}%)  human=${counts.human}`);
  }
  const totalAI = allRows.filter(r => r.label === 1).length;
  const totalHuman = allRows.filter(r => r.label === 0).length;
  console.log(`\n  TOTAL  ${allRows.length}  AI=${totalAI} (${((totalAI/allRows.length)*100).toFixed(1)}%)  human=${totalHuman}`);
}

main().catch(err => { console.error(err); process.exit(1); });
