// Measures old vs new perplexity on AI and human texts, then shows
// how much the weight-0.01 signal actually moves the final composite.

function tokenize(t) { return t.toLowerCase().match(/\b[a-z']+\b/g) || []; }
function getSentences(t) { return t.match(/[^.!?]+[.!?]+/g) || [t]; }

// ── Old formula (single proxy) ─────────────────────────────────────────────
function calcPerplexityOld(text) {
  const words = tokenize(text); if (!words.length) return 0;
  const lengths = words.map(w => w.length);
  const avg = lengths.reduce((a,b)=>a+b,0)/lengths.length;
  const variance = lengths.reduce((a,b)=>a+Math.pow(b-avg,2),0)/lengths.length;
  return Math.max(0, Math.min(100, 100 - (variance * 12)));
}

// ── New ensemble ───────────────────────────────────────────────────────────
function calcPerplexityProxy1(text) { return calcPerplexityOld(text); }

function calcPerplexityProxy2(text) {
  const sentences = getSentences(text);
  if (sentences.length < 3) return { score: 50, dataPoints: sentences.length, confidence: 0 };
  function syllables(word) {
    return word.toLowerCase()
      .replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '')
      .replace(/^y/, '')
      .match(/[aeiouy]{1,2}/g)?.length || 1;
  }
  const sentSyl = sentences.map(s => (s.toLowerCase().match(/\b[a-z']+\b/g)||[]).reduce((sum,w)=>sum+syllables(w),0));
  const avg = sentSyl.reduce((a,b)=>a+b,0)/sentSyl.length;
  if (avg === 0) return { score: 50, dataPoints: sentences.length, confidence: 0 };
  const cv = Math.sqrt(sentSyl.reduce((a,b)=>a+(b-avg)**2,0)/sentSyl.length) / avg;
  return { score: Math.max(0,Math.min(100,Math.round(100-(cv*85)))), dataPoints: sentences.length, confidence: Math.min(sentences.length/10,1.0) };
}

function calcPerplexityProxy3(text) {
  const sentences = getSentences(text);
  if (sentences.length < 5) return { score: 50, confidence: 0, dataPoints: sentences.length, entropy: 0 };
  const lengths = sentences.map(s => (s.trim().match(/\b\w+\b/g) || []).length);
  // 5 bins: 1-7, 8-12, 13-18, 19-25, 26+
  const bins = [0, 0, 0, 0, 0];
  lengths.forEach(l => {
    if      (l <= 7)  bins[0]++;
    else if (l <= 12) bins[1]++;
    else if (l <= 18) bins[2]++;
    else if (l <= 25) bins[3]++;
    else              bins[4]++;
  });
  const total = lengths.length;
  let entropy = 0;
  bins.forEach(b => { if (b > 0) { const p = b/total; entropy -= p * Math.log2(p); } });
  // max = log2(5) ≈ 2.322 bits (perfectly uniform distribution)
  // low entropy = uniform sentence lengths = AI signal
  const maxEntropy = Math.log2(5);
  const score = Math.max(0, Math.min(100, Math.round(100 - (entropy / maxEntropy) * 90)));
  return { score, dataPoints: total, confidence: Math.min(total / 10, 1.0), entropy };
}

function calcPerplexityNew(text) {
  const wc = tokenize(text).length;
  const p1Raw = calcPerplexityProxy1(text);
  const p1 = { score: p1Raw, confidence: 0, dataPoints: wc }; // disabled — always ~0% on all prose
  const p2 = calcPerplexityProxy2(text);
  const p3 = calcPerplexityProxy3(text);
  const valid = [p1,p2,p3].filter(p=>p.confidence>=0.2&&p.dataPoints>=5);
  if (!valid.length) return { score: 50, agreement: 100, validProxyCount: 0 };
  const totalConf = valid.reduce((s,p)=>s+p.confidence,0);
  const ws = valid.reduce((s,p)=>s+p.score*p.confidence,0)/totalConf;
  const stdDev = Math.sqrt(valid.reduce((s,p)=>s+(p.score-ws)**2,0)/valid.length);
  return { score: Math.round(ws), p1: p1.score, p2: p2.score, p3: p3.score,
           agreement: Math.max(0,Math.round(100-stdDev*2)), validProxyCount: valid.length,
           entropy: p3.entropy };
}

// How much does perplexity.score contribute to the composite?
// weight=0.01; evidence = max(0, score-50) * 0.01; composite += evidence * 5
function evidenceDelta(score) { return Math.max(0, score-50) * 0.01 * 5; }

// ── Test texts ─────────────────────────────────────────────────────────────
const samples = [
  { label: 'AI — long formal essay', expected: 'ai',
    text: `Artificial intelligence has emerged as one of the most transformative technologies of the twenty-first century, reshaping industries, redefining labor markets, and challenging longstanding assumptions about human cognition. It is worth noting that the pace of this transformation has accelerated considerably in recent years, driven by advances in deep learning, increased computational power, and the availability of vast datasets. Furthermore, the implications of these developments extend far beyond the realm of computer science, touching virtually every sector of the global economy. The integration of artificial intelligence into the workforce represents one of the most significant economic shifts of our time. It is important to note that this transition is not without precedent; previous technological revolutions, from industrialization to the advent of digital computing, have similarly displaced certain categories of labor while simultaneously creating new forms of employment. However, it is crucial to acknowledge that the speed and scope of AI-driven automation present unique challenges that policymakers and business leaders must carefully navigate. Consequently, educational institutions and training programs must adapt rapidly to equip workers with the skills necessary to thrive in an AI-augmented economy. Moreover, the ethical dimensions of artificial intelligence deployment demand rigorous examination. It is essential to foster robust governance frameworks that ensure AI systems are developed and deployed in ways that respect human dignity and promote equitable outcomes. Stakeholders across government, industry, and civil society must engage in meaningful dialogue to address these pivotal concerns.` },

  { label: 'AI — GPT remote work essay', expected: 'ai',
    text: `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements. It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations. However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout. Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches that address these challenges proactively. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components. In conclusion, remote work represents a paradigm shift in organizational culture that carries both significant opportunities and notable challenges for the modern workforce.` },

  { label: 'AI — academic abstract (GPT)', expected: 'ai',
    text: `This study investigates the relationship between urban green space accessibility and psychological wellbeing outcomes among adult residents of metropolitan areas. Using a mixed-methods approach combining spatial analysis of park distribution data with survey-based measures of self-reported stress, anxiety, and life satisfaction, we examine whether proximity to green space is associated with improved mental health indicators after controlling for relevant socioeconomic covariates. Preliminary findings suggest a statistically significant negative relationship between green space proximity and self-reported stress levels, with effects particularly pronounced among residents in lower-income neighborhoods. Furthermore, qualitative interview data indicate that the perceived quality and safety of green spaces may moderate the relationship between mere proximity and actual usage rates. It is important to note that these findings have implications for urban planning policy and suggest that equitable distribution of high-quality green infrastructure should be considered a public health intervention.` },

  { label: 'HUMAN — personal essay (shoebox)', expected: 'human',
    text: `My grandmother kept every letter anyone ever sent her in a shoebox under her bed. When she died we found forty years of correspondence in there — birthday cards from cousins she'd lost touch with, a handful of letters from my grandfather written during the two years he worked on a fishing boat in Alaska before they got married, notes from neighbors about borrowed things, a postcard from someone named Vivienne postmarked 1978 with no explanation. My grandmother was not a sentimental person in any obvious way. She didn't cry at movies. She gave practical gifts. But she kept every piece of paper anyone had ever handed her, and I've been thinking about that shoebox a lot lately. I'm the kind of person who deletes emails after reading them. Not out of any organizational philosophy — I'm not organized — but because the accumulation makes me anxious. Thousands of messages sitting there unread feels like debt. So I delete, and I don't feel bad about it, and then occasionally I want to find something and can't.` },

  { label: 'HUMAN — technical blog (Node.js)', expected: 'human',
    text: `I spent three days debugging a memory leak that turned out to be six lines of code. Here is what I learned. The app was a Node.js service that processed incoming webhooks and wrote results to a Postgres database. Under load it would gradually eat memory until the process crashed, usually somewhere around the four-hour mark. The first thing I did was add heap snapshots. Nothing useful — the retained objects looked normal. I added more logging. Still nothing obvious. On day two I started reading the ORM source code, which is how I found out that our version of Sequelize had a known issue with connection pool cleanup under certain transaction rollback conditions. Specifically: if a transaction threw an error before the connection was fully established, the connection would be returned to the pool in a weird intermediate state and never properly released. The fix was embarrassingly simple — wrap the transaction initialization in a try-catch and explicitly release the connection on failure. Three days, six lines. I am now a person who reads ORM source code before assuming my own code is wrong.` },

  { label: 'HUMAN — journalism (Card/Krueger)', expected: 'human',
    text: `For decades, the conventional wisdom held that raising the minimum wage would reduce employment — that companies, faced with higher labor costs, would simply hire fewer workers. The argument was intuitive and had the backing of introductory economics textbooks. Then David Card and Alan Krueger published a study in 1994 comparing fast-food employment in New Jersey and Pennsylvania after New Jersey raised its minimum wage. Employment, they found, had actually gone up slightly in New Jersey relative to Pennsylvania. The backlash was fierce. Critics questioned the methodology, the data, the framing. Card himself was essentially pushed out of labor economics for years. He won the Nobel Prize in 2021. What happened in the intervening decades was not just a vindication of one study but a wholesale revision of how economists think about labor markets. The old model assumed workers and employers had roughly equal bargaining power. The newer research suggests that employers, particularly in low-wage industries, often have substantial market power — and that the minimum wage, within limits, corrects for that imbalance rather than destroying jobs.` },

  // Domain-specific edge case — medical writing (long words, should NOT false-positive)
  { label: 'HUMAN — medical writing (domain-specific long words)', expected: 'human',
    text: `The pathophysiology of inflammatory bowel disease involves a dysregulated immune response to commensal microorganisms in genetically susceptible individuals. Both Crohn's disease and ulcerative colitis demonstrate characteristic patterns of mucosal inflammation, though the distribution and histological features differ substantially. In Crohn's disease, transmural granulomatous inflammation may affect any segment of the gastrointestinal tract, producing strictures, fistulae, and abscess formation. Ulcerative colitis, by contrast, involves continuous mucosal and submucosal inflammation restricted to the colon. The therapeutic landscape has expanded considerably with the advent of biologic agents targeting tumour necrosis factor-alpha, interleukin pathways, and integrin-mediated lymphocyte trafficking. Patients refractory to conventional immunomodulatory therapy may benefit from vedolizumab, ustekinumab, or the emerging class of selective Janus kinase inhibitors. Monitoring for opportunistic infections, including reactivation of latent tuberculosis and herpes zoster, remains a critical component of surveillance in immunosuppressed patients. Endoscopic mucosal healing has emerged as a therapeutic target associated with improved long-term outcomes and reduced hospitalisation rates.` },
];

// ── Run ────────────────────────────────────────────────────────────────────
console.log('PERPLEXITY ENSEMBLE — Old vs New, and composite impact\n');
console.log('─'.repeat(82));
console.log(`${'Sample'.padEnd(44)} ${'OLD'.padStart(4)} ${'P1'.padStart(4)} ${'P2'.padStart(4)} ${'P3'.padStart(4)} ${'NEW'.padStart(4)} ${'Δcomp'.padStart(6)}`);
console.log('─'.repeat(82));

samples.forEach(({ label, expected, text }) => {
  const old = calcPerplexityOld(text);
  const nw  = calcPerplexityNew(text);
  const deltaComp = evidenceDelta(nw.score) - evidenceDelta(old);
  const sign = deltaComp > 0.05 ? '+' : deltaComp < -0.05 ? '' : '~';
  console.log(
    `${label.substring(0,43).padEnd(43)} ${String(Math.round(old)+'%').padStart(4)} ` +
    `${String(nw.p1+'%').padStart(4)} ${String(nw.p2+'%').padStart(4)} ${String(nw.p3+'%').padStart(4)} ` +
    `${String(nw.score+'%').padStart(4)} ${(sign+deltaComp.toFixed(2)+'%').padStart(6)} ` +
    `entropy=${nw.entropy?.toFixed(2)}b`
  );
});
console.log('─'.repeat(82));
console.log('\nΔcomp = change in composite score due to perplexity signal alone (weight=0.01)');
console.log('P1=word-length-variance  P2=syllable-variance  P3=sentence-length-entropy');
console.log('\nKey question: does the ensemble fix the medical-writing false positive?');
const med = samples.find(s=>s.label.includes('medical'));
const medOld = calcPerplexityOld(med.text);
const medNew = calcPerplexityNew(med.text);
console.log(`  OLD score: ${Math.round(medOld)}%  (high = would falsely look AI-like)`);
console.log(`  NEW score: ${medNew.score}%  (P1=${medNew.p1} P2=${medNew.p2} P3=${medNew.p3} agreement=${medNew.agreement}%)`);
console.log(`  Entropy:   ${medNew.entropy?.toFixed(3)} bits`);
