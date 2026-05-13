const fs = require('fs');
const path = require('path');

function parseCSVRow(line) {
  const result = []; let current = ''; let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') { if (inQuotes && line[i+1] === '"') { current += '"'; i++; } else { inQuotes = !inQuotes; } }
    else if (ch === ',' && !inQuotes) { result.push(current); current = ''; }
    else { current += ch; }
  }
  result.push(current); return result;
}
function parseCSV(raw) {
  const lines = raw.split('\n').filter(l => l.trim());
  const headers = parseCSVRow(lines[0]);
  return lines.slice(1).map(line => {
    const values = parseCSVRow(line);
    const row = {};
    headers.forEach((h, i) => { row[h.trim()] = values[i]; });
    return row;
  }).filter(r => r['text_content']);
}

function getSentences(text) {
  return (text.match(/[^.!?]+[.!?]+/g) || []).filter(s => s.trim().length > 15);
}

const rows = parseCSV(fs.readFileSync(path.join(__dirname, 'real_dataset.csv'), 'utf8'));

// Sample 5 AI and 5 Human texts — show sentence count and first 200 chars
const aiRows = rows.filter(r => r['is_ai_generated'] === '1').slice(0, 5);
const humanRows = rows.filter(r => r['is_ai_generated'] === '0').slice(0, 5);

console.log('=== AI TEXT SAMPLES ===');
aiRows.forEach((r, i) => {
  const text = r['text_content'];
  const sents = getSentences(text);
  const wordCount = (text.match(/\b\w+\b/g) || []).length;
  console.log(`\n[AI ${i+1}] words=${wordCount} sentences=${sents.length}`);
  console.log('Preview:', text.substring(0, 200));
});

console.log('\n=== HUMAN TEXT SAMPLES ===');
humanRows.forEach((r, i) => {
  const text = r['text_content'];
  const sents = getSentences(text);
  const wordCount = (text.match(/\b\w+\b/g) || []).length;
  console.log(`\n[Human ${i+1}] words=${wordCount} sentences=${sents.length}`);
  console.log('Preview:', text.substring(0, 200));
});

// Overall stats
const allWordCounts = rows.map(r => (r['text_content'].match(/\b\w+\b/g) || []).length);
const allSentCounts = rows.map(r => getSentences(r['text_content']).length);
allWordCounts.sort((a,b) => a-b);
allSentCounts.sort((a,b) => a-b);
const p = (arr, pct) => arr[Math.floor(arr.length * pct)];
console.log('\n=== DATASET STATS ===');
console.log(`Word counts — min:${allWordCounts[0]} p25:${p(allWordCounts,0.25)} median:${p(allWordCounts,0.5)} p75:${p(allWordCounts,0.75)} max:${allWordCounts[allWordCounts.length-1]}`);
console.log(`Sentence counts — min:${allSentCounts[0]} p25:${p(allSentCounts,0.25)} median:${p(allSentCounts,0.5)} p75:${p(allSentCounts,0.75)} max:${allSentCounts[allSentCounts.length-1]}`);
console.log(`Texts with < 4 sentences: ${allSentCounts.filter(s=>s<4).length} / ${allSentCounts.length} (${(allSentCounts.filter(s=>s<4).length/allSentCounts.length*100).toFixed(0)}%)`);
