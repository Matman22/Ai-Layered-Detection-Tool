// Quick sanity-check: does getParagraphs() now split single-paragraph text?

function getSentences(text) {
  return text.match(/[^.!?]+[.!?]+/g) || [text];
}

function getParagraphs(text) {
  const natural = text.split(/\n\s*\n/).filter(p => p.trim().length > 20);
  if (natural.length >= 3) return natural;
  const sentences = getSentences(text).filter(s => s.trim().length > 10);
  if (sentences.length < 6) return natural.length >= 2 ? natural : [text];
  const third = Math.floor(sentences.length / 3);
  return [
    sentences.slice(0, third).join(' '),
    sentences.slice(third, third * 2).join(' '),
    sentences.slice(third * 2).join(' ')
  ].filter(g => g.trim().length > 20);
}

function getNaturalParagraphs(text) {
  return text.split(/\n\s*\n/).filter(p => p.trim().length > 20);
}

const SINGLE_PARA = `Remote work has transformed the way organizations operate. It is worth noting that this shift accelerated following the global pandemic. Furthermore, the implications extend far beyond logistical changes. The nature of workplace culture has been profoundly affected. Workers benefit from the elimination of commutes and greater flexibility. Employers can access a broader talent pool unrestricted by geography. However, communication barriers can emerge across distributed teams. Regular virtual meetings and clear protocols are essential. Providing ergonomic support contributes meaningfully to productivity outcomes.`;

const MULTI_PARA = `Remote work has transformed the way organizations operate. It is worth noting that this shift accelerated following the global pandemic.

Workers benefit from the elimination of commutes and greater flexibility. Employers can access a broader talent pool unrestricted by geography.

However, communication barriers can emerge across distributed teams. Regular virtual meetings and clear protocols are essential.`;

const NO_PUNCT = `remote work has transformed organizations it accelerated after the pandemic workers benefit from no commute employers access broader talent pools however communication barriers emerge across distributed teams regular meetings and clear protocols are essential`;

console.log('=== Single-paragraph text (9 sentences) ===');
const sp = getParagraphs(SINGLE_PARA);
console.log(`getParagraphs: ${sp.length} virtual paragraphs`);
sp.forEach((p, i) => console.log(`  [${i+1}] ${p.substring(0, 70)}... (${p.split(/\s+/).length} words)`));
console.log(`getNaturalParagraphs: ${getNaturalParagraphs(SINGLE_PARA).length} (should be 0 or 1)`);

console.log('\n=== Multi-paragraph text (real breaks) ===');
const mp = getParagraphs(MULTI_PARA);
console.log(`getParagraphs: ${mp.length} paragraphs (should use natural breaks)`);

console.log('\n=== No-punctuation text (our eval dataset case) ===');
const np = getParagraphs(NO_PUNCT);
const ns = getSentences(NO_PUNCT);
console.log(`Sentences detected: ${ns.length}`);
console.log(`getParagraphs: ${np.length} (expected: falls back to full text since <6 sentences)`);
