/**
 * Tests RAIDAR signals on longer, realistically-formatted texts (~400-600 words)
 */

function getSentences(text) {
  return (text.match(/[^.!?]+[.!?]+/g) || []).filter(s => s.trim().length > 15);
}
function calcInterSentenceSimilarity(text) {
  const sentences = getSentences(text);
  if (sentences.length < 4) return 50;
  const sets = sentences.map(s => new Set((s.toLowerCase().match(/\b[a-z]{3,}\b/g) || [])));
  const sims = [];
  for (let i = 0; i < sets.length - 1; i++)
    for (let j = i + 1; j < sets.length; j++) {
      const inter = [...sets[i]].filter(w => sets[j].has(w)).length;
      const union = new Set([...sets[i], ...sets[j]]).size;
      if (union > 0) sims.push(inter / union);
    }
  if (!sims.length) return 50;
  const mean = sims.reduce((a,b)=>a+b,0)/sims.length;
  const std = Math.sqrt(sims.reduce((a,b)=>a+Math.pow(b-mean,2),0)/sims.length);
  return Math.round(Math.min(100, mean * 400) * 0.55 + Math.max(0, 100 - std * 600) * 0.45);
}
function calcSelfBLEU(text) {
  const sentences = getSentences(text);
  if (sentences.length < 3) return 50;
  const tok = sentences.map(s => s.toLowerCase().match(/\b[a-z]+\b/g) || []);
  let total = 0, counted = 0;
  for (let i = 0; i < tok.length; i++) {
    const thisBG = new Set();
    for (let k = 0; k < tok[i].length - 1; k++) thisBG.add(tok[i][k] + ' ' + tok[i][k+1]);
    if (!thisBG.size) continue;
    const otherBG = new Set();
    for (let j = 0; j < tok.length; j++) {
      if (j === i) continue;
      for (let k = 0; k < tok[j].length - 1; k++) otherBG.add(tok[j][k] + ' ' + tok[j][k+1]);
    }
    total += [...thisBG].filter(bg => otherBG.has(bg)).length / thisBG.size;
    counted++;
  }
  return counted ? Math.min(100, Math.round(total / counted * 220)) : 50;
}
function calcVocabPredictability(text) {
  const words = text.toLowerCase().match(/\b[a-z]{4,}\b/g) || [];
  if (words.length < 40) return 50;
  const mid = Math.floor(words.length / 2);
  const first = new Set(words.slice(0, mid));
  const second = new Set(words.slice(mid));
  const inter = [...first].filter(w => second.has(w)).length;
  const union = new Set([...first, ...second]).size;
  return Math.min(100, Math.round(inter / union * 130));
}
function runRAIDARApprox(text) {
  const a = calcInterSentenceSimilarity(text);
  const b = calcSelfBLEU(text);
  const c = calcVocabPredictability(text);
  return { combined: Math.round(a*0.40 + b*0.35 + c*0.25), interSentence: a, selfBLEU: b, vocabPredict: c };
}

const AI_LONG = `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements. The nature of workplace culture, employee productivity, and organizational communication have all been profoundly affected by the widespread adoption of remote work policies.

It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations, reduce overhead costs associated with maintaining physical office spaces, and in many cases observe improvements in employee satisfaction and retention rates.

However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones and lack the informal interactions that naturally occur in shared physical spaces. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout among employees who struggle to disconnect from their professional responsibilities.

Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches that address these challenges proactively. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components of effective remote work frameworks. Additionally, providing employees with appropriate technological tools and ergonomic support for their home offices contributes meaningfully to productivity and wellbeing outcomes.

In conclusion, remote work represents a paradigm shift in organizational culture that carries both significant opportunities and notable challenges. The extent to which organizations leverage these opportunities while mitigating the associated risks will largely determine their competitive positioning in an increasingly distributed global workforce.`;

const HUMAN_LONG = `I've been working from home for three years now, and I still haven't figured out if I like it.

The first few months were genuinely great. No commute meant an extra hour every morning that I used to run or read or just drink my coffee without rushing. My apartment, which had always felt like just a place to sleep, became somewhere I actually wanted to be. I repainted the living room, got a desk that didn't make my back hurt, learned how to make decent pour-over coffee. There was something almost luxurious about all of it.

Then the novelty wore off and the walls started closing in a bit.

The thing nobody tells you about remote work is how much of your social life was invisibly happening at the office. Not even meaningful interaction — just the ambient human presence of other people existing near you. The guy from accounting who always had strong opinions about whatever was in the news. The group lunch on Fridays that you only attended about half the time but still. When that's gone, you don't notice right away, and then one Tuesday you realize you haven't spoken out loud to another person in two days.

I've built some workarounds. I work from coffee shops two mornings a week, not because the wifi is better (it's worse) but because I need to be somewhere with other humans in it. I have a standing call with a friend every Thursday that has nothing to do with work. These aren't solutions exactly, more like maintenance.

My productivity is probably higher now than it was in an office. I get into deep work more easily without the constant interruptions. But I also work later than I used to because the stopping point is less obvious without the physical act of leaving a building. It's a trade-off I'm still figuring out the math on.`;

const samples = [
  { label: 'AI — long formal essay (~430 words)', text: AI_LONG },
  { label: 'Human — long personal essay (~350 words)', text: HUMAN_LONG },
];

samples.forEach(({ label, text }) => {
  const r = runRAIDARApprox(text);
  const words = (text.match(/\b\w+\b/g)||[]).length;
  const sents = getSentences(text).length;
  console.log(`\n[${label}]`);
  console.log(`  words=${words}  sentences=${sents}`);
  console.log(`  Inter-sentence: ${r.interSentence}`);
  console.log(`  Self-BLEU:      ${r.selfBLEU}`);
  console.log(`  Vocab predict:  ${r.vocabPredict}`);
  console.log(`  ► COMBINED:     ${r.combined}`);
});
