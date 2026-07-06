/**
 * Iteration 5 — fetch 5th training source: dmitva/human_ai_generated_text.
 * Each record has human_text + ai_text -> emit two rows (like HC3 handling).
 * Target ~500 rows balanced (250 AI / 250 human), batches of 100, skip <20 words.
 * Output: eval/experiments/source5.csv (text_content,is_ai_generated,source)
 * Usage: node eval/experiments/fetch_source5.js
 */
const https = require('https');
const fs = require('fs');
const path = require('path');

const sleep = ms => new Promise(r => setTimeout(r, ms));
const BATCH_DELAY_MS = 1200;
const BATCH_SIZE = 100;
const PER_CLASS = 250;
const MIN_WORDS = 20;
const SOURCE_NAME = 'dmitva';

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
      process.stdout.write(`\n  [rate-limited] waiting ${wait / 1000}s (retry ${attempt + 1}/${retries})\r`);
      await sleep(wait);
      continue;
    }
    try { return JSON.parse(result.data); }
    catch (e) { throw new Error('JSON parse error: ' + result.data.slice(0, 200)); }
  }
  throw new Error('Rate limit retries exhausted');
}

function escapeCSV(val) {
  if (val === null || val === undefined) return '';
  const str = String(val).replace(/"/g, '""');
  return (str.includes(',') || str.includes('"') || str.includes('\n')) ? `"${str}"` : str;
}

const wordCount = t => t.split(/\s+/).filter(w => w.length > 0).length;

async function main() {
  const human = [], ai = [];
  let offset = 0;
  while ((human.length < PER_CLASS || ai.length < PER_CLASS) && offset < 3000) {
    const result = await fetchRows('dmitva/human_ai_generated_text', 'default', 'train', offset, BATCH_SIZE);
    if (result.error || !result.rows || result.rows.length === 0) break;
    for (const { row } of result.rows) {
      const h = String(row.human_text || '').trim();
      const a = String(row.ai_text || '').trim();
      if (human.length < PER_CLASS && wordCount(h) >= MIN_WORDS) human.push(h);
      if (ai.length < PER_CLASS && wordCount(a) >= MIN_WORDS) ai.push(a);
    }
    offset += BATCH_SIZE;
    process.stdout.write(`  human=${human.length}/${PER_CLASS}  AI=${ai.length}/${PER_CLASS}  (offset ${offset})\r`);
    if (human.length >= PER_CLASS && ai.length >= PER_CLASS) break;
    await sleep(BATCH_DELAY_MS);
  }
  const lines = ['text_content,is_ai_generated,source'];
  for (const t of human) lines.push(`${escapeCSV(t)},0,${SOURCE_NAME}`);
  for (const t of ai) lines.push(`${escapeCSV(t)},1,${SOURCE_NAME}`);
  const outPath = path.join(__dirname, 'source5.csv');
  fs.writeFileSync(outPath, lines.join('\n'), 'utf8');
  console.log(`\nWrote ${human.length} human + ${ai.length} AI rows to ${outPath}`);
}
main().catch(e => { console.error(e); process.exit(1); });
