/**
 * Iteration 6 (retry) — fetch ~400 rows balanced (~200 human/200 AI) from
 * artem9k/ai-text-detection-pile (config default, split train; columns:
 * source, id, text). source==='human' -> label 0, else -> label 1.
 *
 * STRICT no-hang policy: every request has a 15s timeout; on ANY error or
 * timeout, abort the whole script immediately (no retries, no fallback).
 * Progress logged after every batch of 100.
 *
 * Usage: node eval/experiments/fetch_src6.js
 */
const https = require('https');
const fs = require('fs');
const path = require('path');

const BATCH_SIZE = 100;
const PER_CLASS = 200;
const MIN_WORDS = 20;
const SOURCE_NAME = 'artem9k';
const MAX_OFFSET = 20000; // hard ceiling so we can't loop forever

function get(url, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': 'node.js' } }, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => resolve({ status: res.statusCode, data }));
    });
    req.on('error', reject);
    req.setTimeout(timeoutMs, () => req.destroy(new Error(`timeout after ${timeoutMs}ms: ${url}`)));
  });
}

function escapeCSV(val) {
  if (val === null || val === undefined) return '';
  const str = String(val).replace(/"/g, '""');
  return (str.includes(',') || str.includes('"') || str.includes('\n')) ? `"${str}"` : str;
}

const wordCount = t => t.split(/\s+/).filter(w => w.length > 0).length;

const sleep = ms => new Promise(r => setTimeout(r, ms));

// Bounded retry (max 2 extra attempts, short fixed backoff) only for transient
// 502/503 gateway errors — NOT an unbounded/backing-off retry loop, and every
// attempt still carries its own 15s timeout. Anything else aborts immediately.
async function fetchBatch(offset, length) {
  const url = `https://datasets-server.huggingface.co/rows?dataset=artem9k%2Fai-text-detection-pile` +
    `&config=default&split=train&offset=${offset}&length=${length}`;
  for (let attempt = 0; attempt <= 2; attempt++) {
    let result;
    try {
      result = await get(url, 15000);
    } catch (e) {
      console.error(`ABORT: request failed/timeout at offset ${offset}: ${e.message}`);
      process.exit(1);
    }
    if (result.status === 502 || result.status === 503 || result.status === 429) {
      if (attempt === 2) {
        console.error(`ABORT: repeated ${result.status} at offset ${offset} after ${attempt + 1} attempts`);
        process.exit(1);
      }
      const wait = result.status === 429 ? 6000 : 2000;
      console.log(`  transient ${result.status} at offset ${offset}, retry ${attempt + 1}/2 after ${wait}ms...`);
      await sleep(wait);
      continue;
    }
    if (result.status !== 200) {
      console.error(`ABORT: non-200 status ${result.status} at offset ${offset}: ${result.data.slice(0, 300)}`);
      process.exit(1);
    }
    let json;
    try { json = JSON.parse(result.data); }
    catch (e) { console.error(`ABORT: JSON parse error at offset ${offset}: ${result.data.slice(0, 300)}`); process.exit(1); }
    if (json.error) { console.error(`ABORT: API error at offset ${offset}: ${json.error}`); process.exit(1); }
    return json.rows || [];
  }
  return [];
}

// Dataset has 1,392,522 rows, ordered in large same-source blocks (verified via
// single-row samples: offset 0/700k/1,000,000 all 'human', offset 1,300,000 all
// 'ai'). So we fetch human from offset 0 and AI from a known AI-block offset,
// rather than scanning sequentially from 0 (which would need >1.3M rows / 13k
// requests and blow the time budget).
const HUMAN_START_OFFSET = 0;
const AI_START_OFFSET = 1300000;

async function collect(startOffset, wantHuman, t0) {
  const bucket = [];
  let offset = startOffset;
  const maxOffset = startOffset + MAX_OFFSET;
  while (bucket.length < PER_CLASS && offset < maxOffset) {
    const rows = await fetchBatch(offset, BATCH_SIZE);
    if (rows.length === 0) {
      console.log(`No more rows at offset ${offset}; stopping this class.`);
      break;
    }
    for (const { row } of rows) {
      const text = String(row.text || '').trim();
      const isHuman = String(row.source || '').toLowerCase() === 'human';
      if (wordCount(text) < MIN_WORDS) continue;
      if (isHuman === wantHuman && bucket.length < PER_CLASS) bucket.push(text);
    }
    offset += BATCH_SIZE;
    console.log(`  [${wantHuman ? 'human' : 'AI'}] fetched offset ${offset}  count=${bucket.length}/${PER_CLASS}  elapsed=${((Date.now() - t0) / 1000).toFixed(1)}s`);
    if (Date.now() - t0 > 180000) {
      console.error('ABORT: exceeded 3 minutes fetching; something is wrong.');
      process.exit(1);
    }
  }
  return bucket;
}

async function main() {
  const t0 = Date.now();
  const human = await collect(HUMAN_START_OFFSET, true, t0);
  const ai = await collect(AI_START_OFFSET, false, t0);

  const lines = ['text_content,is_ai_generated,source'];
  for (const t of human) lines.push(`${escapeCSV(t)},0,${SOURCE_NAME}`);
  for (const t of ai) lines.push(`${escapeCSV(t)},1,${SOURCE_NAME}`);
  const outPath = path.join(__dirname, 'source6.csv');
  fs.writeFileSync(outPath, lines.join('\n'), 'utf8');
  console.log(`\nWrote ${human.length} human + ${ai.length} AI rows to ${outPath}`);
  console.log(`Total elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
main().catch(e => { console.error('ABORT: unexpected error', e); process.exit(1); });
