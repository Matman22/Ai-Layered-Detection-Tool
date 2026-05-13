/**
 * Extended density melody + full detection test on diverse text samples.
 * Key questions:
 *   1. Does formal human writing (academic, journalistic) get falsely flagged?
 *   2. Does casual-style AI text evade the signal?
 *   3. How does the full scoring model hold up across varied genres?
 *
 * Usage: node eval/test_density_extended.js
 */

// ─── Paste full detection engine inline ───────────────────────────────────────
function tokenize(t) { return t.toLowerCase().match(/\b[a-z']+\b/g) || []; }
function getSentences(t) { return t.match(/[^.!?]+[.!?]+/g) || [t]; }
function getParagraphs(text) {
  const nat = text.split(/\n\s*\n/).filter(p => p.trim().length > 20);
  if (nat.length >= 3) return nat;
  const sents = getSentences(text).filter(s => s.trim().length > 10);
  if (sents.length < 6) return nat.length >= 2 ? nat : [text];
  const t = Math.floor(sents.length / 3);
  return [sents.slice(0,t).join(' '), sents.slice(t,t*2).join(' '), sents.slice(t*2).join(' ')].filter(g => g.trim().length > 20);
}
function getNaturalParagraphs(text) { return text.split(/\n\s*\n/).filter(p => p.trim().length > 20); }

const AI_PHRASES = ['in conclusion','it is worth noting','it is important to note','furthermore',
  'moreover','in summary','to summarize','delve into','it is crucial','in the realm of',
  'as we explore','it becomes evident','navigating','multifaceted','nuanced','at its core',
  'it is essential','foster','leveraging','paradigm','tapestry','underscore','pivotal','utilize',
  "in today's world",'in the context of','underpins','embodies','holistic','robust','streamline',
  'synergy','it goes without saying','needless to say','when it comes to','let us','one can argue',
  'it should be noted','in light of','it is clear that','first and foremost','last but not least'];

function calcBurstiness(t) {
  const s = getSentences(t); if (s.length < 3) return 50;
  const l = s.map(x => x.trim().split(/\s+/).length);
  const avg = l.reduce((a,b)=>a+b,0)/l.length;
  const std = Math.sqrt(l.reduce((a,b)=>a+(b-avg)**2,0)/l.length);
  return Math.max(0, Math.min(100, 100-(std/avg*90)));
}
function calcLexicalDiversity(t) {
  const w = tokenize(t); if (w.length < 10) return 50;
  return Math.max(0, Math.min(100, 100 - (new Set(w).size / w.length * 130)));
}
function calcAIPhrases(t) {
  const lower = t.toLowerCase(); const wc = tokenize(t).length || 1;
  const hits = AI_PHRASES.filter(p => lower.includes(p)).length;
  return Math.min(100, Math.round(hits / wc * 1800));
}
function calcRegisterStability(t) {
  const paras = getParagraphs(t); if (paras.length < 2) return 50;
  const formal = /\b(therefore|consequently|furthermore|nevertheless|predominantly|substantially|implementation|framework|methodology|utilization|paradigm|leverage|facilitate|subsequently|acknowledge|demonstrate|indicate|significant|particular|additional)\b/gi;
  const casual = /\b(really|pretty|kinda|gonna|wanna|stuff|thing|things|lot|lots|bit|bits|yeah|yep|nope|ok|okay|wow|basically|literally|actually|honestly|totally|definitely|probably|maybe|like|just|get|got|very|so|too)\b/gi;
  const scores = paras.map(p => {
    const wc = tokenize(p).length || 1;
    return ((p.match(formal)||[]).length - (p.match(casual)||[]).length * 0.5) / wc * 100;
  });
  const avg = scores.reduce((a,b)=>a+b,0)/scores.length;
  const cv = avg !== 0 ? Math.sqrt(scores.reduce((a,b)=>a+(b-avg)**2,0)/scores.length) / Math.abs(avg) : 1;
  return Math.max(0, Math.min(100, Math.round(80 - cv * 30)));
}
function calcParagraphUniformity(t) {
  const paras = getNaturalParagraphs(t); if (paras.length < 2) return 50;
  const lens = paras.map(p => p.trim().split(/\s+/).length);
  const avg = lens.reduce((a,b)=>a+b,0)/lens.length;
  const cv = avg > 0 ? Math.sqrt(lens.reduce((a,b)=>a+(b-avg)**2,0)/lens.length)/avg : 0;
  return Math.max(0, Math.min(100, Math.round(90 - cv * 80)));
}

// Density melody (full implementation)
const DENSITY_STOPWORDS = new Set([
  'the','a','an','is','are','was','were','be','been','being','am',
  'have','has','had','do','does','did','will','would','could','should',
  'may','might','shall','can','cannot','need','used','let',
  'to','of','in','for','on','with','at','by','from','up','out','off',
  'into','through','during','before','after','above','below','between',
  'about','against','along','around','behind','beside','beyond','despite',
  'except','inside','near','outside','since','toward','under','until',
  'upon','within','without','over','down','back','away','here','there',
  'again','then','once','now','ago','also','even','still','just','very',
  'too','quite','rather','almost','already','always','never','often',
  'usually','sometimes','finally','however','therefore','although',
  'because','unless','whether','though','while','when','where','why','how',
  'all','both','each','few','more','most','other','some','such','many','much',
  'no','nor','not','only','same','so','than','as','if','or','and','but','yet',
  'either','neither','that','this','these','those','it','its','itself',
  'he','she','they','we','you','i','me','him','her','them','us',
  'my','your','his','their','our','whose','what','which','who','whom','any',
  'said','says','say','get','got','go','went','come','came','see','saw',
  'know','knew','think','thought','make','made','take','took','give','gave',
  'look','looked','seem','seemed','become','became','want','wanted','use',
  'find','found','tell','told','ask','asked','work','worked','call','called',
  'try','tried','feel','felt','leave','left','keep','kept','put','set',
  'turn','turned','show','showed','run','ran','move','moved','live','lived',
  'stand','stood','lose','lost','pay','paid','meet','met','sit','sat',
  'also','well','just','like','new','good','old','great','long','little',
  'own','right','big','high','small','large','next','early','young',
  'last','many','part','place','case','week','number','night','point',
  'area','money','fact','month','lot','side','kind','head','house',
  'friend','power','hour','game','line','end','among','name','land',
  'people','world','life','child','day','year','man','woman','way','time',
  'thing','two','three','four','five','six','seven','eight','nine','ten',
]);
function densityWordWeight(w) {
  if (DENSITY_STOPWORDS.has(w)) return 0.05;
  const l = w.length;
  if (l <= 3) return 0.15; if (l <= 6) return 0.40;
  if (l <= 9) return 0.72; return 1.0;
}
function calcDensityMelody(text) {
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length < 80) return { score: 50, reason: 'too short' };
  const wts = words.map(densityWordWeight);
  const W = 25;
  if (words.length < W * 2) return { score: 50, reason: 'too short for curve' };
  const curve = [];
  for (let i = 0; i <= wts.length - W; i++) {
    let s = 0; for (let k = 0; k < W; k++) s += wts[i+k]; curve.push(s/W);
  }
  const n = curve.length;
  const mean = curve.reduce((a,b)=>a+b,0)/n;
  const variance = curve.reduce((a,b)=>a+(b-mean)**2,0)/n;
  const cv = mean > 0.01 ? Math.sqrt(variance)/mean : 0;
  if (mean < 0.32) return { score: 50, mean: mean.toFixed(3), cv: cv.toFixed(3), reason: 'casual vocab — neutral' };
  const highPos = wts.map((w,i)=>({w,i})).filter(({w})=>w>=0.7).map(({i})=>i);
  let gapCV = 0;
  if (highPos.length >= 5) {
    const gaps = []; for (let i=1;i<highPos.length;i++) gaps.push(highPos[i]-highPos[i-1]);
    const gm=gaps.reduce((a,b)=>a+b,0)/gaps.length;
    const gv=gaps.reduce((a,b)=>a+(b-gm)**2,0)/gaps.length;
    gapCV = gm>0 ? Math.sqrt(gv)/gm : 0;
  }
  const smooth = curve.map((_,i)=>{
    const lo=Math.max(0,i-2),hi=Math.min(n-1,i+2);
    let s=0,cnt=0; for(let k=lo;k<=hi;k++){s+=curve[k];cnt++;} return s/cnt;
  });
  const hfEnergy = curve.reduce((s,v,i)=>s+Math.abs(v-smooth[i]),0)/n;
  const cvScore    = Math.max(0,Math.min(100,Math.round(100-(cv-0.10)/0.24*100)));
  const gapCVScore = gapCV>0 ? Math.max(0,Math.min(100,Math.round(100-(gapCV-0.40)/1.0*100))) : 50;
  const hfScore    = Math.max(0,Math.min(100,Math.round(100-(hfEnergy-0.010)/0.040*100)));
  const score = Math.round(cvScore*0.45+gapCVScore*0.30+hfScore*0.25);
  return { score, mean: mean.toFixed(3), cv: cv.toFixed(3), gapCV: gapCV.toFixed(3), hfEnergy: hfEnergy.toFixed(4), cvScore, gapCVScore, hfScore };
}

// ─── Test samples ──────────────────────────────────────────────────────────────
const samples = [

  // ── Formal human writing (the key vulnerability test) ─────────────────────
  {
    label: 'HUMAN — academic paper excerpt (formal, technical)',
    expected: 'human',
    text: `The phenomenon of code-switching has been studied extensively within sociolinguistics, yet its cognitive underpinnings remain incompletely understood. Researchers have proposed several theoretical frameworks to explain why bilingual speakers alternate between languages within a single discourse context. The Matrix Language Frame model, developed by Carol Myers-Scotton, posits that one language serves as the structural backbone of an utterance while the other contributes lexical items. This model has been applied productively to data from numerous language pairs, though critics have noted its difficulty in accounting for instances of symmetric bilingualism where neither language clearly dominates. More recent approaches draw on dynamic systems theory, treating language choice as an emergent property of complex interactions between the speaker, interlocutor, and communicative context. Empirical evidence from neuroimaging studies suggests that code-switching imposes measurable cognitive costs, with greater activation observed in regions associated with executive control when speakers switch compared to when they remain in a single language.`
  },
  {
    label: 'HUMAN — journalism (NYT-style news analysis)',
    expected: 'human',
    text: `For decades, the conventional wisdom held that raising the minimum wage would reduce employment — that companies, faced with higher labor costs, would simply hire fewer workers. The argument was intuitive and had the backing of introductory economics textbooks. Then David Card and Alan Krueger published a study in 1994 comparing fast-food employment in New Jersey and Pennsylvania after New Jersey raised its minimum wage. Employment, they found, had actually gone up slightly in New Jersey relative to Pennsylvania. The backlash was fierce. Critics questioned the methodology, the data, the framing. Card himself was essentially pushed out of labor economics for years. He won the Nobel Prize in 2021. What happened in the intervening decades was not just a vindication of one study but a wholesale revision of how economists think about labor markets. The old model assumed workers and employers had roughly equal bargaining power. The newer research suggests that employers, particularly in low-wage industries, often have substantial market power — and that the minimum wage, within limits, corrects for that imbalance rather than destroying jobs.`
  },
  {
    label: 'HUMAN — college essay (personal statement)',
    expected: 'human',
    text: `Growing up, my grandmother kept a notebook. Not a diary exactly — more like a running ledger of the neighborhood. Who was sick, who had gotten married, who owed whom a favor, whose kid had made it somewhere. She updated it in a cramped cursive I could barely read, in a language I understood only partially. When she died, we found seventeen of these notebooks going back to the late 1960s. My mother couldn't bring herself to throw them away, so they sit in a box in our garage, mostly unread. I think about those notebooks a lot when I think about why I want to study information science. My grandmother was doing something that databases do now — recording, indexing, retrieving, maintaining relationships between entries. But she was also doing something no database does: she cared about the records. Every entry had weight. The guy who owed someone a favor wasn't just a node in a network; he was someone whose reliability she was tracking because it mattered to how she navigated the world.`
  },

  // ── AI writing in casual/conversational style ─────────────────────────────
  {
    label: 'AI — casual explainer (ChatGPT conversational style)',
    expected: 'ai',
    text: `So you want to understand how neural networks actually work? Great, let's break it down in a way that actually makes sense. Think of a neural network like a really big game of telephone, except instead of words getting garbled, information gets refined. At its most basic level, you have inputs — maybe pixel values from an image — and outputs — like "this is a cat." In between, you have layers of nodes that each do a tiny bit of math. Each node takes in some numbers, multiplies them by weights, adds them up, and passes the result through an activation function. The weights are the key thing here. When you train a network, you're essentially adjusting millions of these weights until the network gets good at its task. The way it learns is through something called backpropagation — the network makes a prediction, compares it to the right answer, measures how wrong it was, and then works backward through the layers adjusting weights to make it a little less wrong next time. Do that enough times on enough data and you end up with a network that can recognize cats, translate languages, or generate text.`
  },
  {
    label: 'AI — listicle / productivity article (common AI content farm style)',
    expected: 'ai',
    text: `In today's fast-paced world, managing your time effectively has become more important than ever. Whether you are a student, a professional, or an entrepreneur, developing strong time management skills can significantly impact your productivity and overall wellbeing. It is worth noting that many high-achieving individuals attribute their success not to working harder, but to working smarter. Furthermore, research consistently demonstrates that those who implement structured time management strategies report lower stress levels and higher job satisfaction. One of the most effective approaches involves breaking your day into focused work blocks, a technique popularized by the Pomodoro method. Additionally, prioritizing tasks using frameworks such as the Eisenhower Matrix can help you distinguish between what is urgent and what is truly important. It is important to note that digital distractions represent one of the greatest threats to sustained focus in the modern workplace. Consequently, establishing clear boundaries around technology use during dedicated work periods is essential for maintaining the deep concentration that meaningful work requires. By implementing these evidence-based strategies consistently, you can fundamentally transform your relationship with time and unlock your full potential.`
  },
  {
    label: 'AI — email / professional communication (GPT-drafted)',
    expected: 'ai',
    text: `I hope this message finds you well. I am writing to follow up on our discussion from last week regarding the proposed timeline for the Henderson account deliverables. After careful consideration and consultation with the relevant stakeholders, I wanted to provide an update on where things currently stand. It is important to note that the team has made significant progress on the initial phase of the project, and we are on track to meet the preliminary milestones outlined in our initial planning document. Furthermore, we have identified several opportunities to streamline the workflow that I believe will contribute meaningfully to our overall efficiency. I would like to schedule a brief call at your earliest convenience to discuss these developments in greater detail and ensure that we are aligned on expectations moving forward. Please let me know what times work best for you, and I will do my best to accommodate your schedule. Thank you for your continued support and partnership on this initiative. I look forward to hearing from you.`
  },

  // ── Edge cases ────────────────────────────────────────────────────────────
  {
    label: 'HUMAN — technical blog (developer, informal but dense)',
    expected: 'human',
    text: `I spent three days debugging a memory leak that turned out to be six lines of code. Here is what I learned. The app was a Node.js service that processed incoming webhooks and wrote results to a Postgres database. Under load it would gradually eat memory until the process crashed, usually somewhere around the four-hour mark. The first thing I did was add heap snapshots. Nothing useful — the retained objects looked normal. I added more logging. Still nothing obvious. On day two I started reading the ORM source code, which is how I found out that our version of Sequelize had a known issue with connection pool cleanup under certain transaction rollback conditions. Specifically: if a transaction threw an error before the connection was fully established, the connection would be returned to the pool in a weird intermediate state and never properly released. The fix was embarrassingly simple — wrap the transaction initialization in a try-catch and explicitly release the connection on failure. Three days, six lines. I am now a person who reads ORM source code before assuming my own code is wrong.`
  },
  {
    label: 'HUMAN — Reddit-style personal story (very casual)',
    expected: 'human',
    text: `So this happened to me last month and I still think about it. I was at the airport, running late for a flight, completely stressed out. I get to security and the TSA agent looks at my ID and goes "hey, you dropped something" and hands me back a folded twenty dollar bill. I said thanks and kept moving. Then I'm sitting at the gate and I'm going through my stuff to find my headphones and I realize the twenty wasn't mine. I didn't drop it. He just gave me twenty dollars. I have no idea why. Maybe he found it on the floor. Maybe he was just in a generous mood. I've been tipping better ever since, which I know is not really the lesson here but it's what happened. The random kindness of strangers is a weird thing to receive. You don't know what to do with it except try to pass it on somehow.`
  },
  {
    label: 'AI — scientific abstract style (Claude/GPT academic)',
    expected: 'ai',
    text: `This study investigates the relationship between urban green space accessibility and psychological wellbeing outcomes among adult residents of metropolitan areas. Using a mixed-methods approach combining spatial analysis of park distribution data with survey-based measures of self-reported stress, anxiety, and life satisfaction, we examine whether proximity to green space is associated with improved mental health indicators after controlling for relevant socioeconomic covariates. Our analysis draws on data collected from 2,847 participants across twelve urban districts, with green space accessibility operationalized as the distance to the nearest public park of at least one hectare. Preliminary findings suggest a statistically significant negative relationship between green space proximity and self-reported stress levels, with effects particularly pronounced among residents in lower-income neighborhoods. Furthermore, qualitative interview data indicate that the perceived quality and safety of green spaces may moderate the relationship between mere proximity and actual usage rates. These findings have implications for urban planning policy and suggest that equitable distribution of high-quality green infrastructure should be considered a public health intervention. Future research should employ longitudinal designs to establish causal directionality and examine potential mediating mechanisms.`
  },
];

// ─── Run ──────────────────────────────────────────────────────────────────────
console.log('DENSITY MELODY — extended test\n');
console.log('Checking: formal human writing, casual AI, edge cases\n');
console.log('─'.repeat(74));

let correct = 0, neutral = 0, wrong = 0;

samples.forEach(({ label, expected, text }) => {
  const dm = calcDensityMelody(text);
  const words = (text.match(/\b\w+\b/g)||[]).length;
  const burst = Math.round(calcBurstiness(text));
  const register = Math.round(calcRegisterStability(text));
  const aiPhraseScore = Math.round(calcAIPhrases(text));
  const paraUnif = Math.round(calcParagraphUniformity(text));

  const verdict = dm.score >= 65 ? 'AI-SIGNAL' : dm.score <= 40 ? 'human-signal' : 'neutral';
  const isCorrect = expected === 'ai' ? dm.score >= 55 : dm.score < 55;
  const flag = dm.reason ? '~' : isCorrect ? '✓' : '✗';
  if (dm.reason) neutral++; else if (isCorrect) correct++; else wrong++;

  console.log(`\n[${label}]`);
  console.log(`  words=${words}  burstiness=${burst}%  register=${register}%  ai_phrases=${aiPhraseScore}%  para_unif=${paraUnif}%`);
  if (dm.reason) {
    console.log(`  Density melody: mean=${dm.mean}  CV=${dm.cv}  → ${dm.reason}`);
    console.log(`  ► MELODY: 50% (neutral)  ${flag} [expected: ${expected}]`);
  } else {
    console.log(`  Density melody: mean=${dm.mean}  CV=${dm.cv}  gapCV=${dm.gapCV}  hfEnergy=${dm.hfEnergy}`);
    console.log(`  sub → cv:${dm.cvScore}  gapCV:${dm.gapCVScore}  hf:${dm.hfScore}`);
    const bar = '█'.repeat(Math.round(dm.score/5)).padEnd(20);
    console.log(`  ► MELODY: ${dm.score}%  ${bar}  ${verdict}  ${flag} [expected: ${expected}]`);
  }
});

console.log('\n' + '─'.repeat(74));
console.log(`RESULTS: ${correct} correct  ${neutral} neutral/gated  ${wrong} wrong  (out of ${samples.length})`);
console.log('─'.repeat(74));
