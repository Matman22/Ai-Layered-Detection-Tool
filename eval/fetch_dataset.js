/**
 * Fetches real AI vs Human text samples from HuggingFace's datasets-server REST API.
 * No Python, no login required — public datasets, REST only.
 *
 * Legacy single-dataset mode (unchanged behavior):
 *   node eval/fetch_dataset.js [--rows N]
 *     → writes eval/real_dataset.csv (text_content,is_ai_generated,source_label)
 *
 * Multi-dataset mode (new):
 *   node eval/fetch_dataset.js --multi --per-source 400 --balance
 *     → writes eval/multi_dataset.csv
 *       (text_content,is_ai_generated,source_dataset,domain,model)
 *     → RAID adversarial rows (attack!="none") held out to eval/adversarial_raid.csv
 *
 * Inspect a source's schema before trusting an adapter:
 *   node eval/fetch_dataset.js --peek <sourceKey>
 *
 * Convention: is_ai_generated = 1 for AI, 0 for human.
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const API = 'https://datasets-server.huggingface.co';
const BATCH = 100;               // API hard max per request
const MIN_CHARS = 50;            // skip stub rows
const MAX_REQUESTS_PER_SOURCE = 130; // default safety bound on network paging
const REQUEST_DELAY_MS = 300;    // politeness gap between requests (avoid rate limits)

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ─── HTTP ────────────────────────────────────────────────────────────────────

function getJSON(url, tries = 6) {
  return new Promise((resolve, reject) => {
    const attempt = (n) => {
      https.get(url, { headers: { 'User-Agent': 'node.js' } }, (res) => {
        let data = '';
        res.on('data', (c) => (data += c));
        res.on('end', () => {
          const head = data.trimStart();
          // An HTML body means a rate-limit / error page, not data — back off hard.
          if (head.startsWith('<')) {
            if (n < tries) return setTimeout(() => attempt(n + 1), 2500 * n);
            return reject(new Error('HTML response (rate limited?): ' + head.slice(0, 60)));
          }
          try {
            const json = JSON.parse(data);
            // datasets-server returns 500 + {error} while a dataset is "loading"
            if (json.error && n < tries) return setTimeout(() => attempt(n + 1), 1500 * n);
            resolve(json);
          } catch (e) {
            if (n < tries) return setTimeout(() => attempt(n + 1), 1500 * n);
            reject(new Error('JSON parse error: ' + data.slice(0, 200)));
          }
        });
      }).on('error', (e) => (n < tries ? setTimeout(() => attempt(n + 1), 1500 * n) : reject(e)));
    };
    attempt(1);
  });
}

function rowsUrl(id, config, split, offset, length) {
  return `${API}/rows?dataset=${encodeURIComponent(id)}&config=${encodeURIComponent(config)}` +
         `&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}`;
}

// ─── Dataset registry ──────────────────────────────────────────────────────────
// Each adapter maps a raw API row to our unified schema. `expand` (optional)
// returns an array of unified rows from one raw row (HC3 emits two).

const REGISTRY = {
  // Legacy informal AI-vs-human corpus (kept as a registry entry).
  andy: {
    id: 'andythetechnerd03/AI-human-text', config: 'default', split: 'train',
    textFn: (r) => r.text,
    labelFn: (r) => normLabel(r.generated),
    domainFn: () => '',
    modelFn: () => '',
  },

  // RAID — 11 generators × many domains, plus adversarial attacks.
  // Train only on attack=="none"; attack!="none" is held out (routed by fetcher).
  raid: {
    id: 'liamdugan/raid', config: 'raid', split: 'train',
    textFn: (r) => r.generation,
    labelFn: (r) => (r.model === 'human' ? 0 : 1),
    domainFn: (r) => r.domain || '',
    modelFn: (r) => r.model || '',
    attackFn: (r) => r.attack || 'none',
    large: true,
    maxRequests: 200,  // clean human rows (attack=none, model=human) are ~0.4% — needs more paging
  },

  // MAGE — 27 LLMs, 10 domains. Label MUST come from `src` suffix; the raw
  // integer `label` polarity is inverted vs our convention.
  mage: {
    id: 'yaful/MAGE', config: 'default', split: 'train',
    textFn: (r) => r.text,
    labelFn: (r) => (String(r.src || '').toLowerCase().endsWith('_human') ? 0 : 1),
    domainFn: (r) => String(r.src || '').split('_')[0] || '',
    modelFn: (r) => mageModel(r.src),
    large: true,
  },

  // HC3 — same-question human vs ChatGPT answers; emits two rows per record.
  hc3: {
    id: 'Hello-SimpleAI/HC3', config: 'all', split: 'train',
    large: true,  // rows are grouped by source; random-sample for domain diversity
    expand: (r) => {
      const out = [];
      const h = firstOf(r.human_answers);
      const a = firstOf(r.chatgpt_answers);
      const domain = r.source || '';
      if (h) out.push({ text: h, label: 0, domain, model: '' });
      if (a) out.push({ text: a, label: 1, domain, model: 'chatgpt' });
      return out;
    },
  },
};

function normLabel(v) {
  if (typeof v === 'number') return v ? 1 : 0;
  const s = String(v).toLowerCase();
  return (s === '1' || s === 'ai' || s === 'chatgpt' || s === 'generated' || s === 'fake') ? 1 : 0;
}
function firstOf(v) {
  if (Array.isArray(v)) return v.length ? String(v[0]).trim() : '';
  return v ? String(v).trim() : '';
}
// MAGE src looks like "<domain>_<model>" or "<domain>_human". Pull the model tail.
function mageModel(src) {
  const s = String(src || '');
  if (s.toLowerCase().endsWith('_human')) return '';
  const parts = s.split('_');
  return parts.length > 1 ? parts.slice(1).join('_') : '';
}

// ─── CSV ─────────────────────────────────────────────────────────────────────

function esc(val) {
  if (val === null || val === undefined) return '';
  const s = String(val).replace(/"/g, '""');
  return /[",\n\r]/.test(s) ? `"${s}"` : s;
}

// ─── Row collection ────────────────────────────────────────────────────────────
// Returns unified rows: { text, label, domain, model, attack }.
// Large datasets are sampled across random offset windows for diversity;
// small ones are paged sequentially.

async function collectSource(key, perSource, balance) {
  const a = REGISTRY[key];
  if (!a) throw new Error('Unknown source: ' + key);
  const probe = await getJSON(rowsUrl(a.id, a.config, a.split, 0, 1));
  if (probe.error) throw new Error(`${key}: API error: ${probe.error}`);
  const total = probe.num_rows_total || 0;

  // Adversarial budget (RAID only) sized like a class bucket.
  const wantPerClass = balance ? Math.ceil(perSource / 2) : Infinity;
  const buckets = { 0: [], 1: [] };           // clean rows by label
  const adversarial = { 0: [], 1: [] };       // RAID attack!="none"
  const advCap = a.attackFn ? Math.ceil(perSource / 2) : 0;

  const used = new Set();
  let requests = 0;
  const maxRequests = a.maxRequests || MAX_REQUESTS_PER_SOURCE;
  const useRandom = a.large && total > perSource * 6;

  const enough = () => {
    const cleanDone = balance
      ? (buckets[0].length >= wantPerClass && buckets[1].length >= wantPerClass)
      : (buckets[0].length + buckets[1].length >= perSource);
    const advDone = !a.attackFn ||
      (adversarial[0].length + adversarial[1].length >= advCap * 2) ||
      (adversarial[1].length >= advCap);   // adversarial is mostly AI
    return cleanDone && advDone;
  };

  while (!enough() && requests < maxRequests) {
    let offset;
    if (useRandom) {
      offset = Math.floor(Math.random() * Math.max(1, total - BATCH));
      offset -= offset % BATCH;
      if (used.has(offset)) { if (used.size * BATCH >= total) break; continue; }
    } else {
      offset = requests * BATCH;
      if (offset >= total) break;
    }
    used.add(offset);
    requests++;

    let res;
    try { res = await getJSON(rowsUrl(a.id, a.config, a.split, offset, BATCH)); }
    catch (e) { process.stderr.write(`  [${key}] fetch failed @${offset}: ${e.message}\n`); continue; }
    if (res.error || !Array.isArray(res.rows)) continue;

    for (const { row } of res.rows) {
      const unified = a.expand ? a.expand(row)
        : [{ text: a.textFn(row), label: a.labelFn(row), domain: a.domainFn(row), model: a.modelFn(row) }];
      for (const u of unified) {
        const text = String(u.text || '').trim();
        if (text.length < MIN_CHARS) continue;
        const attack = a.attackFn ? a.attackFn(row) : 'none';
        const rec = { text, label: u.label, domain: u.domain || '', model: u.model || '', attack };
        if (a.attackFn && attack !== 'none') {
          if (adversarial[rec.label].length < advCap) adversarial[rec.label].push(rec);
        } else if (balance) {
          if (buckets[rec.label].length < wantPerClass) buckets[rec.label].push(rec);
        } else if (buckets[0].length + buckets[1].length < perSource) {
          buckets[rec.label].push(rec);
        }
      }
    }
    process.stdout.write(`  [${key}] req ${requests}  clean ${buckets[0].length}H/${buckets[1].length}AI` +
      (a.attackFn ? `  adv ${adversarial[0].length + adversarial[1].length}` : '') + '   \r');
    await sleep(REQUEST_DELAY_MS);
  }
  process.stdout.write('\n');

  const clean = [...buckets[0], ...buckets[1]];
  const adv = [...adversarial[0], ...adversarial[1]];
  return { clean, adv, total, requests };
}

// ─── Confound reporting ──────────────────────────────────────────────────────

function median(nums) {
  if (!nums.length) return 0;
  const s = [...nums].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function reportConfounds(source, rows) {
  const h = rows.filter((r) => r.label === 0).map((r) => r.text.length);
  const ai = rows.filter((r) => r.label === 1).map((r) => r.text.length);
  const mh = median(h), ma = median(ai);
  const cls0 = h.length, cls1 = ai.length;
  let warn = '';
  if (mh > 0 && ma > 0) {
    const ratio = Math.abs(mh - ma) / Math.min(mh, ma);
    if (ratio > 0.25) warn = `  ⚠ LENGTH CONFOUND: medians differ ${(ratio * 100).toFixed(0)}% (classifier may cheat on length)`;
  }
  console.log(`  ${source.padEnd(18)} human=${String(cls0).padStart(4)} ai=${String(cls1).padStart(4)}  ` +
    `medLen H=${mh} AI=${ma}${warn}`);
}

// ─── Multi-dataset main ──────────────────────────────────────────────────────

const MULTI_OUT = path.join(__dirname, 'multi_dataset.csv');
const ADV_OUT = path.join(__dirname, 'adversarial_raid.csv');
const HEADER = 'text_content,is_ai_generated,source_dataset,domain,model';

async function multiMain(perSource, balance, only) {
  // Cheap, label-dense sources first; RAID last (its heavy paging is the most
  // likely to trip rate limits, so don't let it starve the others).
  const sources = only ? [only] : ['andy', 'hc3', 'mage', 'raid'];
  console.log(`Multi-dataset pull: ${sources.join(', ')}  (per-source≈${perSource}, balance=${balance})\n`);

  const clean = [];
  const adv = [];
  for (const key of sources) {
    try {
      const { clean: c, adv: av, total } = await collectSource(key, perSource, balance);
      const tag = key === 'raid' ? 'raid' : key;
      c.forEach((r) => clean.push({ ...r, source: tag }));
      av.forEach((r) => adv.push({ ...r, source: 'raid_adversarial' }));
      console.log(`  [${key}] collected ${c.length} clean + ${av.length} adversarial (of ${total} total)\n`);
    } catch (e) {
      console.error(`  [${key}] FAILED: ${e.message}\n`);
    }
  }

  // Write unified CSV
  const lines = [HEADER];
  for (const r of clean) lines.push([esc(r.text), r.label, esc(r.source), esc(r.domain), esc(r.model)].join(','));
  fs.writeFileSync(MULTI_OUT, lines.join('\n'), 'utf8');

  if (adv.length) {
    const advLines = [HEADER];
    for (const r of adv) advLines.push([esc(r.text), r.label, esc(r.source), esc(r.domain), esc(r.model)].join(','));
    fs.writeFileSync(ADV_OUT, advLines.join('\n'), 'utf8');
  }

  // Summary + confound report
  console.log('=== PER-SOURCE / PER-CLASS SUMMARY ===');
  const bySource = {};
  for (const r of clean) (bySource[r.source] ||= []).push(r);
  for (const [src, rows] of Object.entries(bySource)) reportConfounds(src, rows);
  if (adv.length) reportConfounds('raid_adversarial', adv);

  console.log(`\nSaved ${clean.length} rows → ${MULTI_OUT}`);
  if (adv.length) console.log(`Saved ${adv.length} adversarial rows → ${ADV_OUT}`);
}

// ─── Peek main ───────────────────────────────────────────────────────────────

async function peekMain(key) {
  const a = REGISTRY[key];
  if (!a) { console.error('Unknown source. Keys:', Object.keys(REGISTRY).join(', ')); process.exit(1); }
  console.log(`Peeking ${a.id} (config=${a.config}, split=${a.split})...`);
  const res = await getJSON(rowsUrl(a.id, a.config, a.split, 0, 3));
  if (res.error) { console.error('API error:', res.error); process.exit(1); }
  console.log('num_rows_total:', res.num_rows_total);
  console.log('columns:', (res.features || []).map((f) => f.name).join(', ') ||
    Object.keys(res.rows[0]?.row || {}).join(', '));
  res.rows.forEach(({ row }, i) => {
    console.log(`\n--- raw row ${i} ---`);
    for (const [k, v] of Object.entries(row)) {
      const s = Array.isArray(v) ? `[list len ${v.length}] ${String(v[0]).slice(0, 80)}` : String(v).slice(0, 100);
      console.log(`  ${k}: ${s}`);
    }
    const unified = a.expand ? a.expand(row)
      : [{ text: a.textFn(row), label: a.labelFn(row), domain: a.domainFn(row), model: a.modelFn(row) }];
    unified.forEach((u) => console.log(`  → unified: label=${u.label} domain="${u.domain}" model="${u.model}" text="${String(u.text).slice(0, 60)}..."`));
  });
}

// ─── Legacy single-dataset main (unchanged output contract) ────────────────────

async function legacyMain(totalRows) {
  const a = REGISTRY.andy;
  console.log(`Legacy fetch: ${a.id} (${totalRows} rows) → real_dataset.csv`);
  const csvRows = ['text_content,is_ai_generated,source_label'];
  let fetched = 0;
  for (let offset = 0; offset < totalRows; offset += BATCH) {
    const length = Math.min(BATCH, totalRows - offset);
    const result = await getJSON(rowsUrl(a.id, a.config, a.split, offset, length));
    if (result.error) { console.error('Error at offset', offset, ':', result.error); break; }
    for (const { row } of result.rows) {
      const text = String(a.textFn(row) || '').trim();
      const isAI = a.labelFn(row);
      if (text.length > MIN_CHARS) {
        csvRows.push(`${esc(text)},${isAI},${esc(String(row.generated))}`);
        fetched++;
      }
    }
    process.stdout.write(`  ${fetched}/${totalRows} fetched\r`);
  }
  fs.writeFileSync(path.join(__dirname, 'real_dataset.csv'), csvRows.join('\n'), 'utf8');
  console.log(`\nSaved ${fetched} rows to real_dataset.csv`);
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

function argVal(flag, def) {
  const i = process.argv.indexOf(flag);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

async function main() {
  const argv = process.argv;
  if (argv.includes('--peek')) {
    return peekMain(argVal('--peek', 'andy'));
  }
  if (argv.includes('--multi')) {
    const perSource = parseInt(argVal('--per-source', '400'), 10);
    const balance = argv.includes('--balance');
    const only = argv.includes('--only') ? argVal('--only', null) : null;
    return multiMain(perSource, balance, only);
  }
  // Legacy default
  return legacyMain(parseInt(argVal('--rows', '500'), 10));
}

main().catch((err) => { console.error(err); process.exit(1); });
