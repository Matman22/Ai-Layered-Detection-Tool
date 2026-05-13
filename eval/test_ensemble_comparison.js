/**
 * Compares old single-run density melody formula vs new ensemble.
 * Old formula = word-level sliding window, windowSize=25, threshold=0.32
 * New ensemble = 5 parameter sets (sentence + chunk granularities)
 *
 * Usage: node eval/test_ensemble_comparison.js
 */

// ── Shared infrastructure ─────────────────────────────────────────────────────
const DENSITY_STOPWORDS = new Set(['the','a','an','is','are','was','were','be','been','being','am','have','has','had','do','does','did','will','would','could','should','may','might','shall','can','cannot','need','used','let','to','of','in','for','on','with','at','by','from','up','out','off','into','through','during','before','after','above','below','between','about','against','along','around','behind','beside','beyond','despite','except','inside','near','outside','since','toward','under','until','upon','within','without','over','down','back','away','here','there','again','then','once','now','ago','also','even','still','just','very','too','quite','rather','almost','already','always','never','often','usually','sometimes','finally','however','therefore','although','because','unless','whether','though','while','when','where','why','how','all','both','each','few','more','most','other','some','such','many','much','no','nor','not','only','same','so','than','as','if','or','and','but','yet','either','neither','that','this','these','those','it','its','itself','he','she','they','we','you','i','me','him','her','them','us','my','your','his','their','our','whose','what','which','who','whom','any','said','says','say','get','got','go','went','come','came','see','saw','know','knew','think','thought','make','made','take','took','give','gave','look','looked','seem','seemed','become','became','want','wanted','use','find','found','tell','told','ask','asked','work','worked','call','called','try','tried','feel','felt','leave','left','keep','kept','put','set','turn','turned','show','showed','run','ran','move','moved','live','lived','stand','stood','lose','lost','pay','paid','meet','met','sit','sat','well','like','new','good','old','great','long','little','own','right','big','high','small','large','next','early','young','last','many','part','place','case','week','number','night','point','area','money','fact','month','lot','side','kind','head','house','friend','power','hour','game','line','end','among','name','land','people','world','life','child','day','year','man','woman','way','time','thing','two','three','four','five','six','seven','eight','nine','ten']);

function densityWordWeight(word) {
  const w = word.toLowerCase();
  if (DENSITY_STOPWORDS.has(w)) return 0.05;
  const len = w.length;
  if (len <= 3)  return 0.15;
  if (len <= 6)  return 0.40;
  if (len <= 9)  return 0.72;
  return 1.0;
}

function getSentences(text) {
  return text.match(/[^.!?]+[.!?]+/g) || [text];
}

// ── Old formula (single run, word-level, hardcoded windowSize=25, threshold=0.32) ─
function calcDensityMelodyOld(text) {
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length < 80) return 50;
  const weights = words.map(densityWordWeight);
  const windowSize = 25;
  if (words.length < windowSize * 2) return 50;
  const curve = [];
  for (let i = 0; i <= weights.length - windowSize; i++) {
    let s = 0;
    for (let k = 0; k < windowSize; k++) s += weights[i + k];
    curve.push(s / windowSize);
  }
  const n = curve.length;
  const mean = curve.reduce((a, b) => a + b, 0) / n;
  const variance = curve.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const cv = mean > 0.01 ? Math.sqrt(variance) / mean : 0;
  if (mean < 0.32) return 50;
  const highPos = weights.map((w, i) => w >= 0.7 ? i : -1).filter(i => i >= 0);
  let gapCV = 0;
  if (highPos.length >= 5) {
    const gaps = [];
    for (let i = 1; i < highPos.length; i++) gaps.push(highPos[i] - highPos[i - 1]);
    const gm = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const gv = gaps.reduce((a, b) => a + (b - gm) ** 2, 0) / gaps.length;
    gapCV = gm > 0 ? Math.sqrt(gv) / gm : 0;
  }
  const smooth = curve.map((_, i) => {
    const lo = Math.max(0, i - 2), hi = Math.min(n - 1, i + 2);
    let s = 0, cnt = 0;
    for (let k = lo; k <= hi; k++) { s += curve[k]; cnt++; }
    return s / cnt;
  });
  const hfEnergy = curve.reduce((sum, v, i) => sum + Math.abs(v - smooth[i]), 0) / n;
  const cvScore    = Math.max(0, Math.min(100, Math.round(100 - (cv - 0.10) / 0.24 * 100)));
  const gapCVScore = gapCV > 0 ? Math.max(0, Math.min(100, Math.round(100 - (gapCV - 0.40) / 1.0 * 100))) : 50;
  const hfScore    = Math.max(0, Math.min(100, Math.round(100 - (hfEnergy - 0.010) / 0.040 * 100)));
  return Math.round(cvScore * 0.45 + gapCVScore * 0.30 + hfScore * 0.25);
}

// ── New parameterised formula (shared by both runs and ensemble) ──────────────
function calcDensityMelodyNew(text, { windowSize = 25, threshold = 0.32, granularity = 'word' } = {}) {
  const allWords = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  const curve = [];

  if (granularity === 'sentence') {
    const sents = getSentences(text).filter(s => s.trim().length > 10);
    if (sents.length < windowSize + 1) return { score: 50, dataPoints: 0, confidence: 0 };
    const sentDensities = sents.map(s => {
      const sw = s.toLowerCase().match(/\b[a-z]+\b/g) || [];
      if (!sw.length) return 0;
      return sw.reduce((sum, w) => sum + densityWordWeight(w), 0) / sw.length;
    });
    for (let i = 0; i <= sentDensities.length - windowSize; i++) {
      let sum = 0;
      for (let k = 0; k < windowSize; k++) sum += sentDensities[i + k];
      curve.push(sum / windowSize);
    }
  } else if (granularity === 'chunk') {
    const CHUNK_WORDS = 20;
    const wts = allWords.map(densityWordWeight);
    const chunks = [];
    for (let i = 0; i + CHUNK_WORDS <= wts.length; i += CHUNK_WORDS) {
      chunks.push(wts.slice(i, i + CHUNK_WORDS).reduce((a, b) => a + b, 0) / CHUNK_WORDS);
    }
    if (chunks.length < windowSize + 1) return { score: 50, dataPoints: 0, confidence: 0 };
    for (let i = 0; i <= chunks.length - windowSize; i++) {
      let sum = 0;
      for (let k = 0; k < windowSize; k++) sum += chunks[i + k];
      curve.push(sum / windowSize);
    }
  } else {
    if (allWords.length < 80) return { score: 50, dataPoints: 0, confidence: 0 };
    const wts = allWords.map(densityWordWeight);
    if (allWords.length < windowSize * 2) return { score: 50, dataPoints: 0, confidence: 0 };
    for (let i = 0; i <= wts.length - windowSize; i++) {
      let sum = 0;
      for (let k = 0; k < windowSize; k++) sum += wts[i + k];
      curve.push(sum / windowSize);
    }
  }

  const dataPoints = curve.length;
  if (dataPoints < windowSize * 2) return { score: 50, dataPoints: 0, confidence: 0 };
  const n = dataPoints;
  const mean = curve.reduce((a, b) => a + b, 0) / n;
  if (mean < threshold) return { score: 50, dataPoints: 0, confidence: 0 };
  const variance = curve.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const cv = mean > 0.01 ? Math.sqrt(variance) / mean : 0;
  const highPos = (granularity === 'word')
    ? allWords.map(densityWordWeight).map((w, i) => w >= 0.7 ? i : -1).filter(i => i >= 0)
    : curve.map((v, i) => v > mean ? i : -1).filter(i => i >= 0);
  let gapCV = 0;
  if (highPos.length >= 5) {
    const gaps = [];
    for (let i = 1; i < highPos.length; i++) gaps.push(highPos[i] - highPos[i - 1]);
    const gm = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const gv = gaps.reduce((a, b) => a + (b - gm) ** 2, 0) / gaps.length;
    gapCV = gm > 0 ? Math.sqrt(gv) / gm : 0;
  }
  const smooth = curve.map((_, i) => {
    const lo = Math.max(0, i - 2), hi = Math.min(n - 1, i + 2);
    let s = 0, cnt = 0;
    for (let k = lo; k <= hi; k++) { s += curve[k]; cnt++; }
    return s / cnt;
  });
  const hfEnergy = curve.reduce((sum, v, i) => sum + Math.abs(v - smooth[i]), 0) / n;
  const cvScore    = Math.max(0, Math.min(100, Math.round(100 - (cv - 0.10) / 0.24 * 100)));
  const gapCVScore = gapCV > 0 ? Math.max(0, Math.min(100, Math.round(100 - (gapCV - 0.40) / 1.0 * 100))) : 50;
  const hfScore    = Math.max(0, Math.min(100, Math.round(100 - (hfEnergy - 0.010) / 0.040 * 100)));
  const score = Math.round(cvScore * 0.45 + gapCVScore * 0.30 + hfScore * 0.25);
  const confidence = Math.min(1.0, dataPoints / 15);
  return { score, dataPoints, confidence };
}

function calcDensityMelodyEnsemble(text) {
  const paramSets = [
    { windowSize: 3, threshold: 0.20, granularity: 'sentence' },
    { windowSize: 5, threshold: 0.25, granularity: 'sentence' },
    { windowSize: 7, threshold: 0.30, granularity: 'sentence' },
    { windowSize: 3, threshold: 0.15, granularity: 'chunk' },
    { windowSize: 5, threshold: 0.20, granularity: 'chunk' },
  ];
  const runs = paramSets.map(params => ({ ...calcDensityMelodyNew(text, params), params }));
  const valid = runs.filter(r => r.dataPoints >= 3 && r.confidence > 0);
  if (valid.length === 0) return { score: 50, agreement: 100, validRuns: 0, runs, label: 'LOW CONFIDENCE' };
  const totalWeight = valid.reduce((sum, r) => sum + r.confidence, 0);
  const score = Math.round(valid.reduce((sum, r) => sum + r.score * r.confidence, 0) / totalWeight);
  const scores = valid.map(r => r.score);
  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const stdDev = Math.sqrt(scores.reduce((a, b) => a + (b - meanScore) ** 2, 0) / scores.length);
  const agreement = Math.max(0, Math.round(100 - stdDev));
  const label = agreement >= 75 ? 'HIGH CONFIDENCE' : agreement >= 50 ? 'MODERATE' : 'LOW CONFIDENCE';
  return { score, agreement, validRuns: valid.length, runs, label };
}

// ── Test samples ──────────────────────────────────────────────────────────────
const samples = [
  { label: 'AI — formal essay (heavy AI phrases)', expected: 'AI',
    text: `Artificial intelligence has fundamentally transformed the landscape of modern technology. It is worth noting that the implications of this transformation extend far beyond the realm of computer science. Furthermore, the integration of AI systems into everyday workflows has raised important questions about the future of human labor. It is important to note that these concerns are not without merit. Moreover, the ethical dimensions of AI deployment must be carefully considered by policymakers and technology leaders alike. In conclusion, the responsible development of artificial intelligence requires a nuanced understanding of its multifaceted impact on society. Consequently, stakeholders must engage in meaningful dialogue to address these pivotal challenges.` },

  { label: 'AI — listicle / productivity (content farm)', expected: 'AI',
    text: `In today's fast-paced world, managing your time effectively has become more important than ever. Whether you are a student, a professional, or an entrepreneur, developing strong time management skills can significantly impact your productivity and overall wellbeing. It is worth noting that many high-achieving individuals attribute their success not to working harder, but to working smarter. Furthermore, research consistently demonstrates that those who implement structured time management strategies report lower stress levels and higher job satisfaction. One of the most effective approaches involves breaking your day into focused work blocks, a technique popularized by the Pomodoro method. Additionally, prioritizing tasks using frameworks such as the Eisenhower Matrix can help you distinguish between what is urgent and what is truly important. It is important to note that digital distractions represent one of the greatest threats to sustained focus in the modern workplace. Consequently, establishing clear boundaries around technology use during dedicated work periods is essential for maintaining the deep concentration that meaningful work requires.` },

  { label: 'AI — academic abstract (GPT academic)', expected: 'AI',
    text: `This study investigates the relationship between urban green space accessibility and psychological wellbeing outcomes among adult residents of metropolitan areas. Using a mixed-methods approach combining spatial analysis of park distribution data with survey-based measures of self-reported stress, anxiety, and life satisfaction, we examine whether proximity to green space is associated with improved mental health indicators after controlling for relevant socioeconomic covariates. Our analysis draws on data collected from 2,847 participants across twelve urban districts, with green space accessibility operationalized as the distance to the nearest public park of at least one hectare. Preliminary findings suggest a statistically significant negative relationship between green space proximity and self-reported stress levels, with effects particularly pronounced among residents in lower-income neighborhoods. Furthermore, qualitative interview data indicate that the perceived quality and safety of green spaces may moderate the relationship between mere proximity and actual usage rates. These findings have implications for urban planning policy and suggest that equitable distribution of high-quality green infrastructure should be considered a public health intervention.` },

  { label: 'AI — email draft (professional GPT language)', expected: 'AI',
    text: `I hope this message finds you well. I am writing to follow up on our discussion from last week regarding the proposed timeline for the Henderson account deliverables. After careful consideration and consultation with the relevant stakeholders, I wanted to provide an update on where things currently stand. It is important to note that the team has made significant progress on the initial phase of the project, and we are on track to meet the preliminary milestones outlined in our initial planning document. Furthermore, we have identified several opportunities to streamline the workflow that I believe will contribute meaningfully to our overall efficiency. Moving forward, I would like to schedule a brief call at your earliest convenience to discuss these developments in greater detail.` },

  { label: 'AI — casual explainer (ChatGPT conversational)', expected: 'AI',
    text: `So you want to understand how neural networks actually work? Great, let's break it down in a way that actually makes sense. Think of a neural network like a really big game of telephone, except instead of words getting garbled, information gets refined. At its most basic level, you have inputs — maybe pixel values from an image — and outputs — like "this is a cat." In between, you have layers of nodes that each do a tiny bit of math. Each node takes in some numbers, multiplies them by weights, adds them up, and passes the result through an activation function. The weights are the key thing here. When you train a network, you're essentially adjusting millions of these weights until the network gets good at its task. The way it learns is through something called backpropagation — the network makes a prediction, compares it to the right answer, measures how wrong it was, and then works backward through the layers adjusting weights to make it a little less wrong next time.` },

  { label: 'HUMAN — personal essay (casual)', expected: 'HUMAN',
    text: `I never thought I'd be the kind of person who cares about gardening. Honestly, it started as a joke — my roommate bet me twenty dollars I couldn't keep a cactus alive for three months. I lost that bet spectacularly. But somewhere between killing that first cactus and my fourth failed attempt at growing basil, something clicked. There's something weirdly therapeutic about soil under your fingernails. My mom always had a garden when I was growing up, and I remember thinking it was the most boring thing imaginable. Now I get it. You're not really growing plants, you're just... slowing down. The tomatoes I finally got right last summer tasted completely different from anything from a grocery store.` },

  { label: 'HUMAN — journalism (NYT-style)', expected: 'HUMAN',
    text: `For decades, the conventional wisdom held that raising the minimum wage would reduce employment — that companies, faced with higher labor costs, would simply hire fewer workers. The argument was intuitive and had the backing of introductory economics textbooks. Then David Card and Alan Krueger published a study in 1994 comparing fast-food employment in New Jersey and Pennsylvania after New Jersey raised its minimum wage. Employment, they found, had actually gone up slightly in New Jersey relative to Pennsylvania. The backlash was fierce. Critics questioned the methodology, the data, the framing. Card himself was essentially pushed out of labor economics for years. He won the Nobel Prize in 2021. What happened in the intervening decades was not just a vindication of one study but a wholesale revision of how economists think about labor markets.` },

  { label: 'HUMAN — academic paper (formal human)', expected: 'HUMAN',
    text: `The phenomenon of code-switching has been studied extensively within sociolinguistics, yet its cognitive underpinnings remain incompletely understood. Researchers have proposed several theoretical frameworks to explain why bilingual speakers alternate between languages within a single discourse context. The Matrix Language Frame model, developed by Carol Myers-Scotton, posits that one language serves as the structural backbone of an utterance while the other contributes lexical items. This model has been applied productively to data from numerous language pairs, though critics have noted its difficulty in accounting for instances of symmetric bilingualism where neither language clearly dominates. More recent approaches draw on dynamic systems theory, treating language choice as an emergent property of complex interactions between the speaker, interlocutor, and communicative context.` },

  { label: 'HUMAN — technical blog (developer)', expected: 'HUMAN',
    text: `I spent three days debugging a memory leak that turned out to be six lines of code. Here is what I learned. The app was a Node.js service that processed incoming webhooks and wrote results to a Postgres database. Under load it would gradually eat memory until the process crashed, usually somewhere around the four-hour mark. The first thing I did was add heap snapshots. Nothing useful — the retained objects looked normal. I added more logging. Still nothing obvious. On day two I started reading the ORM source code, which is how I found out that our version of Sequelize had a known issue with connection pool cleanup under certain transaction rollback conditions. The fix was embarrassingly simple — wrap the transaction initialization in a try-catch and explicitly release the connection on failure.` },
];

// ── Run comparison ────────────────────────────────────────────────────────────
const PARAM_LABELS = ['S-w3','S-w5','S-w7','C-w3','C-w5'];

console.log('\nDENSITY MELODY — old single-run vs new ensemble\n');
console.log('─'.repeat(92));
console.log(`${'Sample'.padEnd(44)} ${'OLD'.padStart(5)}  ${'ENS'.padStart(5)}  ${'DELTA'.padStart(6)}  ${'VALID'.padStart(5)}  ${'AGREE'.padStart(5)}  ${'LABEL'}`);
console.log('─'.repeat(92));

const aiDeltas = [];
const humanDeltas = [];

samples.forEach(({ label, expected, text }) => {
  const oldScore = calcDensityMelodyOld(text);
  const ens = calcDensityMelodyEnsemble(text);
  const delta = ens.score - oldScore;
  const deltaStr = (delta >= 0 ? '+' : '') + delta;

  if (expected === 'AI') aiDeltas.push(delta);
  else humanDeltas.push(delta);

  // Individual run scores (only valid ones)
  const runScores = ens.runs.map((r, i) =>
    r.dataPoints >= 3 && r.confidence > 0
      ? `${PARAM_LABELS[i]}:${r.score}`
      : `${PARAM_LABELS[i]}:--`
  ).join(' ');

  const marker = expected === 'AI' ? (ens.score > oldScore ? '↑' : ens.score === oldScore ? '=' : '↓') : '';

  console.log(
    `${label.substring(0,43).padEnd(43)} ` +
    `${String(oldScore).padStart(5)}% ` +
    `${String(ens.score).padStart(5)}% ` +
    `${deltaStr.padStart(6)}  ` +
    `${String(ens.validRuns).padStart(4)}/5  ` +
    `${String(ens.agreement).padStart(4)}%  ` +
    `${ens.label.padEnd(15)} ${marker}`
  );
  console.log(`  runs: ${runScores}`);
});

console.log('─'.repeat(92));

const aiAvgDelta = aiDeltas.length ? (aiDeltas.reduce((a, b) => a + b, 0) / aiDeltas.length).toFixed(1) : 'N/A';
const humanAvgDelta = humanDeltas.length ? (humanDeltas.reduce((a, b) => a + b, 0) / humanDeltas.length).toFixed(1) : 'N/A';
const aiHigher = aiDeltas.filter(d => d > 0).length;
const aiLower  = aiDeltas.filter(d => d < 0).length;
const aiSame   = aiDeltas.filter(d => d === 0).length;

console.log(`\nAI texts    (n=${aiDeltas.length}): avg delta ${aiAvgDelta >= 0 ? '+' : ''}${aiAvgDelta}  ↑ higher:${aiHigher}  = same:${aiSame}  ↓ lower:${aiLower}`);
console.log(`HUMAN texts (n=${humanDeltas.length}): avg delta ${humanAvgDelta >= 0 ? '+' : ''}${humanAvgDelta}`);
console.log('\nLegend: S=sentence, C=chunk, w=windowSize. "--" = run excluded (insufficient data or gated out)');
