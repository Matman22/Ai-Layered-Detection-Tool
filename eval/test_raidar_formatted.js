/**
 * Quick sanity-check of RAIDAR approximation signals on real formatted text.
 * Uses 4 real text samples with proper punctuation.
 */

function getSentences(text) {
  return (text.match(/[^.!?]+[.!?]+/g) || []).filter(s => s.trim().length > 15);
}

function calcInterSentenceSimilarity(text) {
  const sentences = getSentences(text);
  if (sentences.length < 4) return { score: 50, reason: `only ${sentences.length} sentences` };
  const sets = sentences.map(s => new Set((s.toLowerCase().match(/\b[a-z]{3,}\b/g) || [])));
  const similarities = [];
  for (let i = 0; i < sets.length - 1; i++)
    for (let j = i + 1; j < sets.length; j++) {
      const inter = [...sets[i]].filter(w => sets[j].has(w)).length;
      const union = new Set([...sets[i], ...sets[j]]).size;
      if (union > 0) similarities.push(inter / union);
    }
  if (!similarities.length) return { score: 50, reason: 'no pairs' };
  const mean = similarities.reduce((a,b)=>a+b,0) / similarities.length;
  const variance = similarities.reduce((a,b)=>a+Math.pow(b-mean,2),0) / similarities.length;
  const score = Math.round(Math.min(100, mean * 400) * 0.55 + Math.max(0, 100 - Math.sqrt(variance) * 600) * 0.45);
  return { score, mean: mean.toFixed(3), stdDev: Math.sqrt(variance).toFixed(3), sentences: sentences.length };
}

function calcSelfBLEU(text) {
  const sentences = getSentences(text);
  if (sentences.length < 3) return { score: 50, reason: `only ${sentences.length} sentences` };
  const tokenized = sentences.map(s => s.toLowerCase().match(/\b[a-z]+\b/g) || []);
  let total = 0, counted = 0;
  for (let i = 0; i < tokenized.length; i++) {
    const thisBG = new Set();
    for (let k = 0; k < tokenized[i].length - 1; k++) thisBG.add(tokenized[i][k] + ' ' + tokenized[i][k+1]);
    if (!thisBG.size) continue;
    const otherBG = new Set();
    for (let j = 0; j < tokenized.length; j++) {
      if (j === i) continue;
      for (let k = 0; k < tokenized[j].length - 1; k++) otherBG.add(tokenized[j][k] + ' ' + tokenized[j][k+1]);
    }
    total += [...thisBG].filter(bg => otherBG.has(bg)).length / thisBG.size;
    counted++;
  }
  const avg = counted ? total / counted : 0;
  return { score: Math.min(100, Math.round(avg * 220)), avg: avg.toFixed(3) };
}

function calcVocabPredictability(text) {
  const words = text.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
  if (words.length < 40) return { score: 50, reason: 'too short' };
  const mid = Math.floor(words.length / 2);
  const first = new Set(words.slice(0, mid));
  const second = new Set(words.slice(mid));
  const inter = [...first].filter(w => second.has(w)).length;
  const union = new Set([...first, ...second]).size;
  const jaccard = union > 0 ? inter / union : 0;
  return { score: Math.min(100, Math.round(jaccard * 130)), jaccard: jaccard.toFixed(3) };
}

function calcEntropyUniformity(text) {
  const sentences = getSentences(text);
  if (sentences.length < 4) return { score: 50, reason: `only ${sentences.length} sentences` };
  const allWords = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  const freq = {};
  allWords.forEach(w => { freq[w] = (freq[w]||0)+1; });
  const total = allWords.length;
  const entropies = sentences.map(s => {
    const sw = s.toLowerCase().match(/\b[a-z]+\b/g) || [];
    if (!sw.length) return null;
    return -sw.reduce((sum, w) => sum + (freq[w]||1)/total * Math.log2((freq[w]||1)/total), 0) / sw.length;
  }).filter(e => e !== null);
  if (entropies.length < 3) return { score: 50 };
  const mean = entropies.reduce((a,b)=>a+b,0)/entropies.length;
  const variance = entropies.reduce((a,b)=>a+Math.pow(b-mean,2),0)/entropies.length;
  return { score: Math.max(0, Math.min(100, Math.round(85 - Math.sqrt(variance) * 280))), stdDev: Math.sqrt(variance).toFixed(3) };
}

function runRAIDARApprox(text) {
  const interSentence = calcInterSentenceSimilarity(text);
  const selfBLEU      = calcSelfBLEU(text);
  const vocabPredict  = calcVocabPredictability(text);
  const entropyUnif   = calcEntropyUniformity(text);
  const combined = Math.round(interSentence.score * 0.35 + selfBLEU.score * 0.25 + vocabPredict.score * 0.25 + entropyUnif.score * 0.15);
  return { combined, interSentence, selfBLEU, vocabPredict, entropyUnif };
}

// ─── Test samples ─────────────────────────────────────────────────────────────

const samples = [
  {
    label: 'AI — formal essay (GPT-style)',
    text: `Artificial intelligence has fundamentally transformed the landscape of modern technology. It is worth noting that the implications of this transformation extend far beyond the realm of computer science. Furthermore, the integration of AI systems into everyday workflows has raised important questions about the future of human labor. It is important to note that these concerns are not without merit. Moreover, the ethical dimensions of AI deployment must be carefully considered by policymakers and technology leaders alike. In conclusion, the responsible development of artificial intelligence requires a nuanced understanding of its multifaceted impact on society. Consequently, stakeholders must engage in meaningful dialogue to address these pivotal challenges. The landscape of AI governance is evolving rapidly, and it is essential to foster collaboration between diverse perspectives.`
  },
  {
    label: 'Human — personal essay (natural)',
    text: `I never thought I'd be the kind of person who cares about gardening. Honestly, it started as a joke — my roommate bet me twenty dollars I couldn't keep a cactus alive for three months. I lost that bet spectacularly. But somewhere between killing that first cactus and my fourth failed attempt at growing basil, something clicked. There's something weirdly therapeutic about soil under your fingernails. My mom always had a garden when I was growing up, and I remember thinking it was the most boring thing imaginable. Now I get it. You're not really growing plants, you're just... slowing down. The tomatoes I finally got right last summer tasted completely different from anything from a grocery store. Not necessarily better — just more real somehow.`
  },
  {
    label: 'AI — informational text (GPT-style)',
    text: `Climate change represents one of the most significant challenges facing humanity in the twenty-first century. The scientific consensus indicates that global temperatures have risen approximately 1.1 degrees Celsius above pre-industrial levels. This warming trend has been attributed primarily to anthropogenic greenhouse gas emissions, particularly carbon dioxide and methane. Consequently, the frequency and intensity of extreme weather events have increased substantially. Furthermore, rising sea levels pose an existential threat to low-lying coastal regions around the world. It is crucial to understand that the impacts of climate change are not distributed equally across populations. Vulnerable communities in developing nations face disproportionate risks despite contributing minimally to global emissions. Therefore, addressing climate change requires both mitigation strategies to reduce emissions and adaptation measures to protect at-risk populations.`
  },
  {
    label: 'Human — blog post (casual)',
    text: `So I finally tried that new ramen place downtown everyone keeps talking about. Went on a Tuesday thinking it'd be quiet — nope, still a 40-minute wait. Worth it though? Kind of? The broth was genuinely great, really deep and rich, the kind you can tell took forever. But my noodles were slightly overcooked and the chashu was a bit dry. My friend got the spicy miso and she was raving about it the whole walk home, so maybe I just ordered wrong. The space itself is tiny and the tables are practically on top of each other — overheard at least three conversations I did not ask to be part of. Would I go back? Probably. Would I wait 40 minutes again? Ask me when I'm not cold and hungry.`
  },
  {
    label: 'AI — academic paragraph (Claude-style)',
    text: `The relationship between social media usage and adolescent mental health has garnered significant scholarly attention in recent years. Several longitudinal studies have demonstrated correlations between high social media consumption and elevated rates of anxiety and depression among teenagers. However, it is important to note that correlation does not imply causation, and researchers have highlighted the complexity of these relationships. The mechanisms through which social media may influence mental health outcomes remain a subject of ongoing investigation. Social comparison theory provides one useful framework for understanding these dynamics, suggesting that frequent exposure to idealized representations of others' lives may negatively impact self-esteem. Additionally, the displacement of sleep and physical activity by screen time may contribute to poorer mental health outcomes independently of social media content itself.`
  }
];

// ─── Run and display ──────────────────────────────────────────────────────────
samples.forEach(({ label, text }) => {
  const r = runRAIDARApprox(text);
  const sents = getSentences(text).length;
  const words = (text.match(/\b\w+\b/g) || []).length;
  console.log(`\n[${ label }]`);
  console.log(`  words=${words}  sentences=${sents}`);
  console.log(`  Inter-sentence:    ${r.interSentence.score}  ${JSON.stringify(r.interSentence)}`);
  console.log(`  Self-BLEU:         ${r.selfBLEU.score}  ${JSON.stringify(r.selfBLEU)}`);
  console.log(`  Vocab predict:     ${r.vocabPredict.score}  ${JSON.stringify(r.vocabPredict)}`);
  console.log(`  Entropy uniformity:${r.entropyUnif.score}  ${JSON.stringify(r.entropyUnif)}`);
  console.log(`  ► COMBINED:        ${r.combined}`);
});
