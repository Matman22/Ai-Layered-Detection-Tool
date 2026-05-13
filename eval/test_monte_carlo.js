/**
 * Monte Carlo sampling analysis — standalone test.
 * Runs runMonteCarloAnalysis on all test texts (400+ words only).
 * Usage: node eval/test_monte_carlo.js
 */

// ── Core utilities ────────────────────────────────────────────────────────────
function tokenize(t) { return t.toLowerCase().match(/\b[a-z']+\b/g) || []; }
function getSentences(t) { return t.match(/[^.!?]+[.!?]+/g) || [t]; }
function getParagraphs(text) {
  const nat = text.split(/\n\s*\n/).filter(p=>p.trim().length>20);
  if (nat.length>=3) return nat;
  const sents=getSentences(text).filter(s=>s.trim().length>10);
  if (sents.length<6) return nat.length>=2?nat:[text];
  const th=Math.floor(sents.length/3);
  return [sents.slice(0,th).join(' '),sents.slice(th,th*2).join(' '),sents.slice(th*2).join(' ')].filter(g=>g.trim().length>20);
}
function getNaturalParagraphs(t) { return t.split(/\n\s*\n/).filter(p=>p.trim().length>20); }

// ── Detection functions ───────────────────────────────────────────────────────
function calcBurstiness(t) {
  const s=getSentences(t); if(s.length<3) return 50;
  const l=s.map(x=>x.trim().split(/\s+/).length);
  const avg=l.reduce((sum,b)=>sum+b,0)/l.length;  // use distinct name to avoid shadowing in reducer
  const std=Math.sqrt(l.reduce((sum,b)=>sum+(b-avg)**2,0)/l.length);
  return Math.max(0,Math.min(100,100-(std/avg*90)));
}
const AI_PHRASES_T1=['it is worth noting','it is important to note',"it's worth noting",'it should be noted','it is important to understand','it is clear that','it becomes evident','it goes without saying','needless to say','delve into','delves into','multifaceted','at its core','in the realm of','as we explore','tapestry','underscore','pivotal','embodies','underpins','first and foremost','last but not least',"in today's world","in today's fast-paced","in today's ever-changing",'one can argue','it is crucial to note','it is essential to note'];
const AI_PHRASES_T2=['in conclusion','furthermore','moreover','in summary','to summarize','it is crucial','it is essential','in the context of','leveraging','foster','paradigm','robust','streamline','synergy','utilize','navigating','holistic','nuanced','in light of','when it comes to','let us','consequently','in addition','moving forward','going forward','best practices','game-changer','cutting-edge','proactive','actionable','impactful','transformative','stakeholders'];
function calcAIPhrases(t) {
  const l=t.toLowerCase();
  const t1=AI_PHRASES_T1.filter(p=>l.includes(p)).length;
  const t2=AI_PHRASES_T2.filter(p=>l.includes(p)).length;
  const t1s=t1===0?0:Math.min(95,58+(t1-1)*12);
  const t2s=t2===0?0:t2===1?15:Math.min(55,27+(t2-2)*8);
  const combo=t1>=1&&t2>=2?12:t1>=1&&t2>=1?5:0;
  return { score: Math.min(100,t1s+t2s+combo) };
}
function calcFormalityShift(t) {
  const p=getParagraphs(t); if(p.length<2) return 50;
  const fm=/\b(therefore|consequently|furthermore|nevertheless|predominantly|substantially|implementation|framework|methodology|utilization|paradigm|leverage|facilitate|subsequently|acknowledge|demonstrate|indicate|significant|particular|additional)\b/gi;
  const cs=/\b(really|pretty|kinda|gonna|wanna|stuff|thing|things|lot|lots|bit|bits|yeah|yep|nope|ok|okay|wow|basically|literally|actually|honestly|totally|definitely|probably|maybe|like|just|get|got|very|so|too)\b/gi;
  const sc=p.map(x=>{const wc=tokenize(x).length||1;return((x.match(fm)||[]).length-(x.match(cs)||[]).length*0.5)/wc*100;});
  const avg=sc.reduce((s,y)=>s+y,0)/sc.length;
  const cv=avg!==0?Math.sqrt(sc.reduce((s,y)=>s+(y-avg)**2,0)/sc.length)/Math.abs(avg):1;
  return Math.max(0,Math.min(100,Math.round(80-cv*30)));
}
function calcTransitions(t) {
  const tw=['however','therefore','furthermore','moreover','consequently','additionally','nevertheless','subsequently','in contrast','on the other hand','in addition','as a result','for example','for instance','in fact','indeed','meanwhile','likewise','similarly','in conclusion','to summarize','overall'];
  const w=tokenize(t).length||1;
  const c=tw.reduce((a,p)=>a+(t.toLowerCase().match(new RegExp('\\b'+p.replace(' ','\\s+')+'\\b','g'))||[]).length,0);
  return Math.min(100,Math.round(c/w*800));
}
function calcVocabClustering(t) {
  const p=getParagraphs(t); if(p.length<2) return 50;
  const aw=tokenize(t);
  const freq={};
  aw.forEach(w=>{if(w.length>4) freq[w]=(freq[w]||0)+1;});
  const kt=new Set(Object.keys(freq).filter(w=>freq[w]>=2));
  if(kt.size===0) return 30;
  const d=p.map(x=>{const pw=tokenize(x);return pw.filter(w=>kt.has(w)).length/pw.length;});
  const avg=d.reduce((s,y)=>s+y,0)/d.length;
  const cv=avg>0?Math.sqrt(d.reduce((s,y)=>s+(y-avg)**2,0)/d.length)/avg:0;
  return Math.max(0,Math.min(100,85-(cv*75)));
}

// ── Monte Carlo functions ─────────────────────────────────────────────────────
function sampleTextSubset(text, windowSize) {
  const words = text.split(/\s+/).filter(w=>w.length>0);
  if (words.length <= windowSize) return text;
  const maxStart = words.length - windowSize;
  const startIdx = Math.floor(Math.random() * (maxStart + 1));
  return words.slice(startIdx, startIdx + windowSize).join(' ');
}

function runMonteCarloAnalysis(text) {
  const words = text.split(/\s+/).filter(w=>w.length>0);
  const wc = words.length;

  let sampleCount, windowPct;
  if (wc < 200)       { sampleCount = 8;  windowPct = 0.60; }
  else if (wc < 500)  { sampleCount = 12; windowPct = 0.40; }
  else if (wc < 1000) { sampleCount = 16; windowPct = 0.35; }
  else                { sampleCount = 20; windowPct = 0.25; }

  const windowSize = Math.max(50, Math.round(wc * windowPct));
  if (windowSize >= wc) {
    return { score:50, mean:50, stdDev:0, range:0, consistency:100, mixed:false,
             classification:'INCONCLUSIVE', heatmap:new Array(10).fill(null),
             samples:[], validSamples:0 };
  }

  const sampleResults = [];
  for (let i = 0; i < sampleCount; i++) {
    const maxStart = wc - windowSize;
    const startWord = Math.floor(Math.random() * (maxStart + 1));
    const subset = words.slice(startWord, startWord + windowSize).join(' ');
    const startPct = Math.round((startWord / wc) * 100);
    const b  = calcBurstiness(subset);
    const a  = calcAIPhrases(subset).score;
    const f  = calcFormalityShift(subset);
    const tr = calcTransitions(subset);
    const vc = calcVocabClustering(subset);
    const avg = Math.round((b + a + f + tr + vc) / 5);
    sampleResults.push({ score: avg, startPct });
  }

  const sScores = sampleResults.map(r => r.score);
  const mean = Math.round(sScores.reduce((a,b)=>a+b,0) / sScores.length);
  const variance = sScores.reduce((a,b)=>a+(b-mean)**2,0) / sScores.length;
  const stdDev = Math.round(Math.sqrt(variance));
  const range = Math.max(...sScores) - Math.min(...sScores);
  const consistency = Math.max(0, Math.min(100, 100 - stdDev * 2));
  const mixed = range > 35 && stdDev > 18;

  let classification;
  if (mixed)                                 classification = 'MIXED ORIGIN';
  else if (consistency >= 75 && mean >= 50)  classification = 'CONSISTENTLY AI';
  else if (consistency >= 75 && mean <= 38)  classification = 'CONSISTENTLY HUMAN';
  else                                       classification = 'INCONCLUSIVE';

  const heatmap = new Array(10).fill(null);
  const heatCounts = new Array(10).fill(0), heatSums = new Array(10).fill(0);
  for (const r of sampleResults) {
    const bucket = Math.min(9, Math.floor(r.startPct / 10));
    heatSums[bucket] += r.score; heatCounts[bucket]++;
  }
  for (let i = 0; i < 10; i++) {
    if (heatCounts[i] > 0) heatmap[i] = Math.round(heatSums[i] / heatCounts[i]);
  }

  return { score: mean, mean, stdDev, range, consistency, mixed,
           classification, heatmap, samples: sampleResults, validSamples: sampleResults.length };
}

// ── Test samples ──────────────────────────────────────────────────────────────
const samples = [
  // ── AI texts ──────────────────────────────────────────────────────────────
  { label: 'AI — long formal essay (600+ words)', expected: 'ai',
    text: `Artificial intelligence has emerged as one of the most transformative technologies of the twenty-first century, reshaping industries, redefining labor markets, and challenging longstanding assumptions about human cognition. It is worth noting that the pace of this transformation has accelerated considerably in recent years, driven by advances in deep learning, increased computational power, and the availability of vast datasets. Furthermore, the implications of these developments extend far beyond the realm of computer science, touching virtually every sector of the global economy.

The integration of artificial intelligence into the workforce represents one of the most significant economic shifts of our time. It is important to note that this transition is not without precedent; previous technological revolutions, from industrialization to the advent of digital computing, have similarly displaced certain categories of labor while simultaneously creating new forms of employment. However, it is crucial to acknowledge that the speed and scope of AI-driven automation present unique challenges that policymakers and business leaders must carefully navigate. Consequently, educational institutions and training programs must adapt rapidly to equip workers with the skills necessary to thrive in an AI-augmented economy.

Moreover, the ethical dimensions of artificial intelligence deployment demand rigorous examination. The development of algorithmic systems capable of making consequential decisions in domains such as healthcare, criminal justice, and financial services raises profound questions about accountability, transparency, and fairness. It is essential to foster robust governance frameworks that ensure AI systems are developed and deployed in ways that respect human dignity and promote equitable outcomes. Stakeholders across government, industry, and civil society must engage in meaningful dialogue to address these pivotal concerns before the technology becomes further entrenched in critical infrastructure.

The geopolitical dimensions of artificial intelligence competition also merit careful consideration. In the context of an increasingly multipolar world, leading nations are investing heavily in AI research and development as a strategic priority. It is clear that the nation or coalition that achieves sustained leadership in artificial intelligence capabilities will likely enjoy significant advantages in economic productivity, military capability, and diplomatic influence. This dynamic has prompted calls for international cooperation to establish norms and standards governing the development and deployment of AI systems, particularly in sensitive domains such as autonomous weapons and surveillance technology.

Looking forward, the trajectory of artificial intelligence development will be shaped by a complex interplay of technical, economic, social, and political factors. It is essential to approach this challenge with a nuanced understanding of both the opportunities and risks that advanced AI systems present. Holistic frameworks that consider not only technical performance metrics but also broader societal impacts will be necessary to guide responsible innovation. The paradigm shift we are witnessing underscores the importance of maintaining a long-term perspective, prioritizing human wellbeing alongside technological progress, and ensuring that the benefits of artificial intelligence are distributed equitably across society. In conclusion, the path forward requires sustained collaboration among diverse stakeholders committed to leveraging AI's transformative potential while safeguarding the values and institutions that underpin democratic societies.` },

  { label: 'AI — academic abstract (GPT formal)', expected: 'ai',
    text: `This study investigates the relationship between urban green space accessibility and psychological wellbeing outcomes among adult residents of metropolitan areas. Using a mixed-methods approach combining spatial analysis of park distribution data with survey-based measures of self-reported stress, anxiety, and life satisfaction, we examine whether proximity to green space is associated with improved mental health indicators after controlling for relevant socioeconomic covariates. Our analysis draws on data collected from 2,847 participants across twelve urban districts, with green space accessibility operationalized as the distance to the nearest public park of at least one hectare. Preliminary findings suggest a statistically significant negative relationship between green space proximity and self-reported stress levels, with effects particularly pronounced among residents in lower-income neighborhoods. Furthermore, qualitative interview data indicate that the perceived quality and safety of green spaces may moderate the relationship between mere proximity and actual usage rates. It is important to note that these findings have implications for urban planning policy and suggest that equitable distribution of high-quality green infrastructure should be considered a public health intervention. Stakeholders in city planning must leverage these insights to foster more holistic approaches to urban wellbeing.` },

  { label: 'AI — multi-paragraph GPT essay (remote work)', expected: 'ai',
    text: `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements.

It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations.

However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout.

Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches that address these challenges proactively. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components. In conclusion, remote work represents a paradigm shift in organizational culture that carries both significant opportunities and notable challenges.` },

  // ── Human texts ───────────────────────────────────────────────────────────
  { label: 'HUMAN — long personal essay (shoebox letters, 600+w)', expected: 'human',
    text: `My grandmother kept every letter anyone ever sent her in a shoebox under her bed. When she died we found forty years of correspondence in there — birthday cards from cousins she'd lost touch with, a handful of letters from my grandfather written during the two years he worked on a fishing boat in Alaska before they got married, notes from neighbors about borrowed things, a postcard from someone named Vivienne postmarked 1978 with no explanation. My grandmother was not a sentimental person in any obvious way. She didn't cry at movies. She gave practical gifts. But she kept every piece of paper anyone had ever handed her, and I've been thinking about that shoebox a lot lately.

I'm the kind of person who deletes emails after reading them. Not out of any organizational philosophy — I'm not organized — but because the accumulation makes me anxious. Thousands of messages sitting there unread feels like debt. So I delete, and I don't feel bad about it, and then occasionally I want to find something and can't, and I feel briefly bad, and then I forget about it. My digital life is full of these small erasures.

The thing is, I don't actually know what I've lost. That's the nature of deletion: you can't audit the archive you didn't keep. My grandmother could go back through that shoebox and tell you exactly who had been in her life and when and how. There's a card in there from a woman named Patricia that reads only "Thank you for everything, you'll never know how much." My grandmother is gone now and I'll never know what that meant, but I know Patricia existed and that something happened between them that mattered.

I spent an afternoon with those letters after the funeral, reading through them with my aunt. What struck me most wasn't the content exactly but the texture of the handwriting — how you could see my grandfather getting tired toward the end of a long letter, or how my grandmother's friend Carol always drew a small star at the end of her sentences instead of periods. Nobody does that in email. There's no equivalent of the slightly crooked stamp or the coffee ring on the envelope. Digital communication is clean in a way that real life isn't.

I know there's a version of this essay where I conclude that we should all write more letters and preserve more things and be more intentional about memory. But I don't think that's quite right either. My grandmother wasn't trying to be intentional. She just didn't throw things away. The archive was accidental, which might be why it feels so honest.

What I think I'm actually mourning when I think about the shoebox isn't the letters themselves but the evidence they provide that people move through your life and leave marks. Patricia thanked my grandmother for something my grandmother apparently never told anyone about. That exists somewhere in the record now. Most of what I do doesn't exist anywhere. I send a message, the other person reads it, we have an exchange, I delete it, they probably delete it, and it's as though it never happened — except that it did, except that we are slightly different people because it happened, except that the mark is there even if the paper isn't.

That's probably fine. I'm not going to start keeping shoeboxes. But I do think about Vivienne and her unexplained postcard from 1978, and I wonder what I'm not leaving behind.` },

  { label: 'HUMAN — college essay (information science)', expected: 'human',
    text: `Growing up, my grandmother kept a notebook. Not a diary exactly — more like a running ledger of the neighborhood. Who was sick, who had gotten married, who owed whom a favor, whose kid had made it somewhere. She updated it in a cramped cursive I could barely read, in a language I understood only partially. When she died, we found seventeen of these notebooks going back to the late 1960s. My mother couldn't bring herself to throw them away, so they sit in a box in our garage, mostly unread. I think about those notebooks a lot when I think about why I want to study information science. My grandmother was doing something that databases do now — recording, indexing, retrieving, maintaining relationships between entries. But she was also doing something no database does: she cared about the records. Every entry had weight. The guy who owed someone a favor wasn't just a node in a network; he was someone whose reliability she was tracking because it mattered to how she navigated the world. There's a version of data science that treats people as rows in a table. My grandmother's version treated people as people. I want to spend my career figuring out the gap between those two things.` },

  { label: 'HUMAN — journalism (Card/Krueger min wage)', expected: 'human',
    text: `For decades, the conventional wisdom held that raising the minimum wage would reduce employment — that companies, faced with higher labor costs, would simply hire fewer workers. The argument was intuitive and had the backing of introductory economics textbooks. Then David Card and Alan Krueger published a study in 1994 comparing fast-food employment in New Jersey and Pennsylvania after New Jersey raised its minimum wage. Employment, they found, had actually gone up slightly in New Jersey relative to Pennsylvania. The backlash was fierce. Critics questioned the methodology, the data, the framing. Card himself was essentially pushed out of labor economics for years. He won the Nobel Prize in 2021. What happened in the intervening decades was not just a vindication of one study but a wholesale revision of how economists think about labor markets. The old model assumed workers and employers had roughly equal bargaining power. The newer research suggests that employers, particularly in low-wage industries, often have substantial market power — and that the minimum wage, within limits, corrects for that imbalance rather than destroying jobs.` },

  { label: 'HUMAN — opinion piece (Philadelphia habituate)', expected: 'human',
    text: `There's a particular kind of exhaustion that comes from living somewhere you used to love. I've been in Philadelphia for six years now, and somewhere around year four I stopped noticing the things that made me move here in the first place — the weird little bars, the fact that you can actually afford an apartment with a second bedroom, the way the city kind of just lets you be. Now I mostly notice the potholes and the fact that my gym raised prices again.

I don't think this is a Philadelphia problem. I think it's a me problem, or maybe just a humans-in-general problem. We habituate. The good stuff fades into background and the annoyances stay sharp because annoyances require action. You can't just ignore a pothole the same way you stop seeing a mural.

What I've been trying is embarrassingly simple: I've started walking different routes. Not for exercise, just to see different blocks. Last week I ended up on a street I'd somehow never been on despite living eight blocks away. There was a guy repairing clocks in a window. I don't know why that hit me the way it did, but it kind of reset something.` },

  { label: 'HUMAN — technical blog (Node.js memory leak)', expected: 'human',
    text: `I spent three days debugging a memory leak that turned out to be six lines of code. Here is what I learned. The app was a Node.js service that processed incoming webhooks and wrote results to a Postgres database. Under load it would gradually eat memory until the process crashed, usually somewhere around the four-hour mark. The first thing I did was add heap snapshots. Nothing useful — the retained objects looked normal. I added more logging. Still nothing obvious. On day two I started reading the ORM source code, which is how I found out that our version of Sequelize had a known issue with connection pool cleanup under certain transaction rollback conditions. Specifically: if a transaction threw an error before the connection was fully established, the connection would be returned to the pool in a weird intermediate state and never properly released. The fix was embarrassingly simple — wrap the transaction initialization in a try-catch and explicitly release the connection on failure. Three days, six lines. I am now a person who reads ORM source code before assuming my own code is wrong.` },
];

// ── Run ───────────────────────────────────────────────────────────────────────
console.log('MONTE CARLO ANALYSIS — AI vs Human texts\n');
console.log('─'.repeat(78));
console.log(`${'Sample'.padEnd(46)} ${'wc'.padStart(4)}  ${'mean'.padStart(5)}  ${'σ'.padStart(4)}  ${'range'.padStart(5)}  ${'consist'.padStart(7)}  Classification`);
console.log('─'.repeat(78));

// Run 3 times and average to reduce randomness variance
const RUNS = 3;
samples.forEach(({ label, expected, text }) => {
  const wc = tokenize(text).length;
  if (wc < 150) { console.log(`  SKIP (${wc}w)  ${label}`); return; }

  // Average across RUNS to stabilize
  const results = Array.from({length: RUNS}, () => runMonteCarloAnalysis(text));
  const mean     = Math.round(results.reduce((a,r)=>a+r.mean,0) / RUNS);
  const stdDev   = Math.round(results.reduce((a,r)=>a+r.stdDev,0) / RUNS);
  const range    = Math.round(results.reduce((a,r)=>a+r.range,0) / RUNS);
  const consist  = Math.round(results.reduce((a,r)=>a+r.consistency,0) / RUNS);
  // Pick most common classification across runs
  const clsCounts = {}; results.forEach(r=>{clsCounts[r.classification]=(clsCounts[r.classification]||0)+1;});
  const cls = Object.entries(clsCounts).sort((a,b)=>b[1]-a[1])[0][0];

  const correct = expected === 'ai'
    ? (mean >= 50 || cls === 'CONSISTENTLY AI' || cls === 'MIXED ORIGIN')
    : (mean < 50 || cls === 'CONSISTENTLY HUMAN');
  const flag = correct ? '✓' : '✗';

  console.log(`${flag} ${label.substring(0,44).padEnd(44)} ${String(wc).padStart(4)}  ${String(mean+'%').padStart(5)}  ${String(stdDev+'%').padStart(4)}  ${String(range+'%').padStart(5)}  ${String(consist+'%').padStart(7)}  ${cls}`);
});

console.log('\n─'.repeat(78));
console.log('Each row is averaged over 3 independent MC runs to reduce sampling noise.');
console.log('mean: avg AI score across all samples  |  σ: spread  |  consist: 100-(σ*2)');
console.log('Classification: CONSISTENTLY AI / CONSISTENTLY HUMAN / MIXED ORIGIN / INCONCLUSIVE');
