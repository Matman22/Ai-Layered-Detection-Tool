/**
 * Fetches real AI vs Human text samples from HuggingFace datasets-server API
 * No Python, no login required — public dataset, REST API.
 * Usage: node eval/fetch_dataset.js [--rows N]
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const DATASET = 'andythetechnerd03/AI-human-text';
const OUT_PATH = path.join(__dirname, 'real_dataset.csv');

const rowArg = process.argv.indexOf('--rows');
const TOTAL_ROWS = rowArg !== -1 ? parseInt(process.argv[rowArg + 1]) : 500;
const BATCH_SIZE = 100; // API max per request

function fetchRows(offset, length) {
  return new Promise((resolve, reject) => {
    const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}&config=default&split=train&offset=${offset}&length=${length}`;
    https.get(url, { headers: { 'User-Agent': 'node.js' } }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error('JSON parse error: ' + data.substring(0, 200))); }
      });
    }).on('error', reject);
  });
}

function escapeCSV(val) {
  if (val === null || val === undefined) return '';
  const str = String(val).replace(/"/g, '""');
  return str.includes(',') || str.includes('"') || str.includes('\n') ? `"${str}"` : str;
}

async function main() {
  // First: peek at schema
  console.log('Fetching schema from HuggingFace API...');
  const peek = await fetchRows(0, 3);

  if (peek.error) {
    console.error('API error:', peek.error);
    process.exit(1);
  }

  const sampleRow = peek.rows[0]?.row || {};
  console.log('Columns:', Object.keys(sampleRow).join(', '));
  console.log('Sample row:');
  Object.entries(sampleRow).forEach(([k, v]) => {
    const val = String(v);
    console.log(`  ${k}: ${val.substring(0, 120)}${val.length > 120 ? '...' : ''}`);
  });
  console.log();

  // Detect text and label columns
  const cols = Object.keys(sampleRow);
  const textCol = cols.find(c => /text|content|prompt|essay/i.test(c));
  const labelCol = cols.find(c => /label|class|generated|ai|human|source/i.test(c));

  if (!textCol || !labelCol) {
    console.error('Could not auto-detect text/label columns. Columns found:', cols);
    process.exit(1);
  }

  console.log(`Using: text="${textCol}", label="${labelCol}"`);

  // Check label values
  const labelVals = peek.rows.map(r => r.row[labelCol]);
  console.log('Sample label values:', labelVals);
  console.log();

  // Fetch full dataset in batches
  console.log(`Fetching ${TOTAL_ROWS} rows in batches of ${BATCH_SIZE}...`);
  const csvRows = ['text_content,is_ai_generated,source_label'];
  let fetched = 0;

  for (let offset = 0; offset < TOTAL_ROWS; offset += BATCH_SIZE) {
    const length = Math.min(BATCH_SIZE, TOTAL_ROWS - offset);
    const result = await fetchRows(offset, length);

    if (result.error) {
      console.error('Error at offset', offset, ':', result.error);
      break;
    }

    for (const { row } of result.rows) {
      const text = String(row[textCol] || '').trim();
      const rawLabel = row[labelCol];

      // Normalize label to 0/1
      let isAI;
      if (typeof rawLabel === 'number') {
        isAI = rawLabel;
      } else {
        const ls = String(rawLabel).toLowerCase();
        isAI = (ls === '1' || ls === 'ai' || ls === 'chatgpt' || ls === 'generated' || ls === 'fake') ? 1 : 0;
      }

      if (text.length > 50) { // skip empty/stub rows
        csvRows.push(`${escapeCSV(text)},${isAI},${escapeCSV(String(rawLabel))}`);
        fetched++;
      }
    }

    process.stdout.write(`  ${fetched}/${TOTAL_ROWS} fetched\r`);
  }

  fs.writeFileSync(OUT_PATH, csvRows.join('\n'), 'utf8');
  console.log(`\nSaved ${fetched} rows to ${OUT_PATH}`);

  // Label distribution
  const aiCount = csvRows.slice(1).filter(r => r.endsWith(',1,') || r.match(/,1,"?[^"]*"?$/)).length;
  console.log(`  AI: ~${aiCount} | Human: ~${fetched - aiCount}`);
}

main().catch(err => { console.error(err); process.exit(1); });
