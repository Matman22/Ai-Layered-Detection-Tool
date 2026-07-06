/**
 * Iteration 6 (retry) — SINGLE probe of splits, then SINGLE probe of 5 rows,
 * for artem9k/ai-text-detection-pile. Hard 15s timeout on each request, no
 * retries, no fallback datasets. Abort immediately on any failure/timeout.
 */
const https = require('https');

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

async function main() {
  const t0 = Date.now();
  console.log('Step 1: splits probe...');
  const splitsUrl = 'https://datasets-server.huggingface.co/splits?dataset=artem9k%2Fai-text-detection-pile';
  let splitsResult;
  try {
    splitsResult = await get(splitsUrl, 15000);
  } catch (e) {
    console.error(`ABORT: splits probe failed/timeout: ${e.message}`);
    process.exit(1);
  }
  if (splitsResult.status !== 200) {
    console.error(`ABORT: splits probe non-200 status ${splitsResult.status}: ${splitsResult.data.slice(0, 300)}`);
    process.exit(1);
  }
  let splitsJson;
  try { splitsJson = JSON.parse(splitsResult.data); }
  catch (e) { console.error(`ABORT: splits probe JSON parse error: ${splitsResult.data.slice(0, 300)}`); process.exit(1); }
  console.log(JSON.stringify(splitsJson, null, 2));
  const first = splitsJson.splits && splitsJson.splits[0];
  if (!first) { console.error('ABORT: no splits returned'); process.exit(1); }
  console.log(`\nUsing config=${first.config} split=${first.split}`);
  console.log(`Elapsed after splits probe: ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  console.log('\nStep 2: rows probe (5 rows)...');
  const rowsUrl = `https://datasets-server.huggingface.co/rows?dataset=artem9k%2Fai-text-detection-pile&config=${encodeURIComponent(first.config)}&split=${encodeURIComponent(first.split)}&offset=0&length=5`;
  let rowsResult;
  try {
    rowsResult = await get(rowsUrl, 15000);
  } catch (e) {
    console.error(`ABORT: rows probe failed/timeout: ${e.message}`);
    process.exit(1);
  }
  if (rowsResult.status !== 200) {
    console.error(`ABORT: rows probe non-200 status ${rowsResult.status}: ${rowsResult.data.slice(0, 300)}`);
    process.exit(1);
  }
  let rowsJson;
  try { rowsJson = JSON.parse(rowsResult.data); }
  catch (e) { console.error(`ABORT: rows probe JSON parse error: ${rowsResult.data.slice(0, 300)}`); process.exit(1); }
  if (!rowsJson.rows || !rowsJson.rows.length) { console.error('ABORT: no rows returned'); process.exit(1); }
  console.log(`Columns: ${Object.keys(rowsJson.rows[0].row).join(', ')}`);
  const oneHuman = rowsJson.rows.find(r => String(r.row.source || '').toLowerCase() === 'human');
  const oneAi = rowsJson.rows.find(r => String(r.row.source || '').toLowerCase() !== 'human');
  console.log('\n--- Sample predicted HUMAN row ---');
  console.log(JSON.stringify(oneHuman ? oneHuman.row : rowsJson.rows[0], null, 2).slice(0, 1500));
  console.log('\n--- Sample predicted AI row ---');
  console.log(JSON.stringify(oneAi ? oneAi.row : rowsJson.rows[1], null, 2).slice(0, 1500));
  console.log(`\nTotal elapsed: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}
main().catch(e => { console.error('ABORT: unexpected error', e); process.exit(1); });
