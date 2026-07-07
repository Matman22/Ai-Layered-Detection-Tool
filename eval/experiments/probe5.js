// Probe candidate 5th sources via HF datasets-server.
const https = require('https');
function get(url) {
  return new Promise((resolve, reject) => {
    https.get(url, { headers: { 'User-Agent': 'node.js' } }, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => resolve({ status: res.statusCode, data: d }));
    }).on('error', reject);
  });
}
async function main() {
  const ds = process.argv[2];
  const cfg = process.argv[3];
  const enc = encodeURIComponent(ds);
  if (!cfg) {
    const r = await get(`https://datasets-server.huggingface.co/splits?dataset=${enc}`);
    console.log('SPLITS status', r.status);
    console.log(r.data.slice(0, 2000));
    return;
  }
  const r = await get(`https://datasets-server.huggingface.co/rows?dataset=${enc}&config=${encodeURIComponent(cfg)}&split=${process.argv[4] || 'train'}&offset=${process.argv[5] || 0}&length=5`);
  console.log('ROWS status', r.status);
  try {
    const j = JSON.parse(r.data);
    if (j.error) { console.log('ERROR:', j.error); return; }
    console.log('num_rows_total:', j.num_rows_total);
    console.log('columns:', j.features.map(f => f.name).join(', '));
    for (const { row } of j.rows.slice(0, 3)) {
      for (const [k, v] of Object.entries(row)) {
        const s = String(v);
        console.log(`  ${k}: ${s.length > 300 ? s.slice(0, 300) + '…' : s}`);
      }
      console.log('  ---');
    }
  } catch (e) { console.log(r.data.slice(0, 500)); }
}
main().catch(e => { console.error(e); process.exit(1); });
