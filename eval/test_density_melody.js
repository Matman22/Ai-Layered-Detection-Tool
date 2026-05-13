/**
 * Lexical Density Melody — prototype and calibration test
 *
 * Concept: assign each word an information weight based on frequency tier,
 * build a rolling density curve across the document, then measure how
 * "flat" or "irregular" that curve is.
 *
 * AI text: evenly distributed information throughout → flat melody, low CV
 * Human text: density spikes at core arguments, flattens at transitions → irregular melody, high CV
 *
 * Usage: node eval/test_density_melody.js
 */

// ─── Word frequency tiers ──────────────────────────────────────────────────────
// Top ~250 function words and ultra-common content words → near-zero weight
const STOPWORDS = new Set([
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
  'my','your','his','her','their','our','its','whose',
  'what','which','who','whom','any','another','every','certain','own',
  'said','says','say','get','got','go','went','come','came','see','saw',
  'know','knew','think','thought','make','made','take','took','give','gave',
  'look','looked','seem','seemed','become','became','want','wanted','use','used',
  'find','found','tell','told','ask','asked','work','worked','call','called',
  'try','tried','feel','felt','leave','left','keep','kept','put','set',
  'turn','turned','show','showed','run','ran','move','moved','live','lived',
  'stand','stood','lose','lost','pay','paid','meet','met','sit','sat',
  'also','well','just','like','new','good','old','great','long','little',
  'own','right','big','high','small','large','next','early','young','important',
  'public','private','real','best','free','able','need','sure','true','hard',
  'last','many','part','place','case','week','company','system','program',
  'question','government','number','night','point','home','water','room','mother',
  'area','money','story','fact','month','lot','right','study','book','eye',
  'job','word','business','issue','side','kind','head','house','service',
  'friend','father','power','hour','game','line','end','among','while','name',
  'land','help','hand','people','world','life','child','day','year','man','woman',
  'way','time','thing','two','three','four','five','six','seven','eight','nine','ten',
]);

function wordWeight(word) {
  const w = word.toLowerCase();
  if (STOPWORDS.has(w)) return 0.05;
  const len = w.length;
  if (len <= 3)  return 0.15; // short non-stop words: "run", "big" etc.
  if (len <= 6)  return 0.40; // common content words: "table", "write"
  if (len <= 9)  return 0.72; // domain content words: "analysis", "network"
  return 1.0;                  // specialist/academic: "fundamentally", "multifaceted"
}

// ─── Density curve ─────────────────────────────────────────────────────────────
function buildDensityCurve(text, windowSize = 25) {
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length < windowSize * 2) return null;

  const weights = words.map(wordWeight);

  // Sliding window average — each position is mean weight of 25-word span
  const curve = [];
  for (let i = 0; i <= weights.length - windowSize; i++) {
    let sum = 0;
    for (let k = 0; k < windowSize; k++) sum += weights[i + k];
    curve.push(sum / windowSize);
  }
  return { curve, wordCount: words.length, weights };
}

// ─── Melody metrics ────────────────────────────────────────────────────────────
function calcDensityMelody(text) {
  const result = buildDensityCurve(text, 25);
  if (!result || result.curve.length < 15) return { score: 50, reason: 'text too short' };

  const { curve } = result;
  const n = curve.length;

  // Basic stats on the curve
  const mean = curve.reduce((a, b) => a + b, 0) / n;
  const variance = curve.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const cv = mean > 0.01 ? std / mean : 0; // coefficient of variation

  // Peak-to-trough: how dramatic is the range?
  // ── Gate: signal is only meaningful for formal/content-rich text ──────────────
  // Casual human writing uses short low-weight words uniformly → flat curve too,
  // but for a different reason than AI. We only fire the signal when the text is
  // formal enough that "flatness" is meaningful evidence.
  // From calibration data: AI formal texts have mean 0.42-0.48, human casual 0.21-0.29.
  if (mean < 0.32) {
    return { score: 50, mean: mean.toFixed(3), cv: cv.toFixed(3), reason: 'casual vocabulary — melody neutral' };
  }

  // ── Gap CV: how regularly are high-information words spaced? ──────────────────
  // AI: evenly distributed domain vocabulary → small gaps between high-weight words,
  //     and those gaps are consistent (low CV of gaps).
  // Human: high-weight words burst together at argument peaks, then sparse in transitions
  //        → irregular gap sizes (high CV of gaps).
  const highWeightPos = result.weights
    .map((w, i) => ({ w, i }))
    .filter(({ w }) => w >= 0.7)
    .map(({ i }) => i);

  let gapCV = 0;
  if (highWeightPos.length >= 5) {
    const gaps = [];
    for (let i = 1; i < highWeightPos.length; i++) gaps.push(highWeightPos[i] - highWeightPos[i - 1]);
    const gapMean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const gapVar  = gaps.reduce((a, b) => a + (b - gapMean) ** 2, 0) / gaps.length;
    gapCV = gapMean > 0 ? Math.sqrt(gapVar) / gapMean : 0;
    // High gapCV = bursty/irregular spacing = human signal
    // Low gapCV  = even spacing = AI signal
  }

  // ── High-frequency energy: how much rapid local variation in the curve? ────────
  // Smooth the curve with a 5-point moving average, measure deviation from smoothed.
  // AI flat curve: small deviation from smoothed (low HF energy)
  // Human spiky curve: larger deviation (higher HF energy)
  const smooth = [];
  for (let i = 0; i < n; i++) {
    const lo = Math.max(0, i - 2), hi = Math.min(n - 1, i + 2);
    let s = 0, cnt = 0;
    for (let k = lo; k <= hi; k++) { s += curve[k]; cnt++; }
    smooth.push(s / cnt);
  }
  const hfEnergy = curve.reduce((sum, v, i) => sum + Math.abs(v - smooth[i]), 0) / n;

  // ── Score formula ───────────────────────────────────────────────────────────────
  // Both metrics: lower value = flatter = more AI-like → higher score
  //
  // CV of density curve:
  //   Calibrated range for formal text: AI ~0.135-0.145 (very flat), human formal ~0.20-0.25+
  //   Score: 100 at CV=0.10, 50 at CV=0.22, 0 at CV=0.34
  const cvScore  = Math.max(0, Math.min(100, Math.round(100 - (cv - 0.10) / 0.24 * 100)));

  // Gap CV of high-weight words:
  //   AI: regular spacing → gapCV ~0.5-0.8
  //   Human: bursty spacing → gapCV ~1.0-1.5+
  //   Score: 100 at gapCV=0.4, 50 at gapCV=0.9, 0 at gapCV=1.4
  const gapCVScore = gapCV > 0
    ? Math.max(0, Math.min(100, Math.round(100 - (gapCV - 0.40) / 1.0 * 100)))
    : 50;

  // HF energy:
  //   AI: ~0.015-0.025 (smooth), Human: ~0.025-0.045 (rougher)
  //   Score: 100 at hfEnergy=0.010, 50 at hfEnergy=0.030, 0 at hfEnergy=0.050
  const hfScore = Math.max(0, Math.min(100, Math.round(100 - (hfEnergy - 0.010) / 0.040 * 100)));

  const score = Math.round(cvScore * 0.45 + gapCVScore * 0.30 + hfScore * 0.25);

  return {
    score,
    mean:      mean.toFixed(3),
    cv:        cv.toFixed(3),
    gapCV:     gapCV.toFixed(3),
    hfEnergy:  hfEnergy.toFixed(4),
    cvScore, gapCVScore, hfScore,
    std:       std.toFixed(3),
    curveLen:  n,
  };
}

// ─── Test samples ──────────────────────────────────────────────────────────────
const samples = [
  {
    label: 'AI — formal essay (GPT-style, single paragraph)',
    text: `Artificial intelligence has fundamentally transformed the landscape of modern technology. It is worth noting that the implications of this transformation extend far beyond the realm of computer science. Furthermore, the integration of AI systems into everyday workflows has raised important questions about the future of human labor. It is important to note that these concerns are not without merit. Moreover, the ethical dimensions of AI deployment must be carefully considered by policymakers and technology leaders alike. In conclusion, the responsible development of artificial intelligence requires a nuanced understanding of its multifaceted impact on society. Consequently, stakeholders must engage in meaningful dialogue to address these pivotal challenges. The landscape of AI governance is evolving rapidly, and it is essential to foster collaboration between diverse perspectives.`,
  },
  {
    label: 'Human — personal essay (casual, single paragraph)',
    text: `I never thought I'd be the kind of person who cares about gardening. Honestly, it started as a joke — my roommate bet me twenty dollars I couldn't keep a cactus alive for three months. I lost that bet spectacularly. But somewhere between killing that first cactus and my fourth failed attempt at growing basil, something clicked. There's something weirdly therapeutic about soil under your fingernails. My mom always had a garden when I was growing up, and I remember thinking it was the most boring thing imaginable. Now I get it. You're not really growing plants, you're just... slowing down. The tomatoes I finally got right last summer tasted completely different from anything from a grocery store. Not necessarily better — just more real somehow.`,
  },
  {
    label: 'AI — multi-paragraph essay (with real breaks)',
    text: `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements.

It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations.

However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout.

Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components of effective remote work frameworks.`,
  },
  {
    label: 'Human — blog post (casual, multi-paragraph)',
    text: `So I finally tried that new ramen place downtown everyone keeps talking about. Went on a Tuesday thinking it'd be quiet — nope, still a 40-minute wait. Worth it though? Kind of?

The broth was genuinely great, really deep and rich, the kind you can tell took forever. But my noodles were slightly overcooked and the chashu was a bit dry. My friend got the spicy miso and she was raving about it the whole walk home, so maybe I just ordered wrong.

The space itself is tiny and the tables are practically on top of each other — overheard at least three conversations I did not ask to be part of. Would I go back? Probably. Would I wait 40 minutes again? Ask me when I'm not cold and hungry.`,
  },
  {
    label: 'AI — long formal essay (600+ words, multi-paragraph)',
    text: `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements. The nature of workplace culture, employee productivity, and organizational communication have all been profoundly affected by the widespread adoption of remote work policies.

It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations, reduce overhead costs associated with maintaining physical office spaces, and in many cases observe improvements in employee satisfaction and retention rates.

However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones and lack the informal interactions that naturally occur in shared physical spaces. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout among employees who struggle to disconnect from their professional responsibilities.

Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches that address these challenges proactively. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components of effective remote work frameworks. Additionally, providing employees with appropriate technological tools and ergonomic support for their home offices contributes meaningfully to productivity and wellbeing outcomes.

In conclusion, remote work represents a paradigm shift in organizational culture that carries both significant opportunities and notable challenges. The extent to which organizations leverage these opportunities while mitigating the associated risks will largely determine their competitive positioning in an increasingly distributed global workforce.`,
  },
  {
    label: 'Human — long personal essay (600+ words, multi-paragraph)',
    text: `I've been working from home for three years now, and I still haven't figured out if I like it.

The first few months were genuinely great. No commute meant an extra hour every morning that I used to run or read or just drink my coffee without rushing. My apartment, which had always felt like just a place to sleep, became somewhere I actually wanted to be. I repainted the living room, got a desk that didn't make my back hurt, learned how to make decent pour-over coffee. There was something almost luxurious about all of it.

Then the novelty wore off and the walls started closing in a bit.

The thing nobody tells you about remote work is how much of your social life was invisibly happening at the office. Not even meaningful interaction — just the ambient human presence of other people existing near you. The guy from accounting who always had strong opinions about whatever was in the news. The group lunch on Fridays that you only attended about half the time but still. When that's gone, you don't notice right away, and then one Tuesday you realize you haven't spoken out loud to another person in two days.

I've built some workarounds. I work from coffee shops two mornings a week, not because the wifi is better (it's worse) but because I need to be somewhere with other humans in it. I have a standing call with a friend every Thursday that has nothing to do with work. These aren't solutions exactly, more like maintenance.

My productivity is probably higher now than it was in an office. I get into deep work more easily without the constant interruptions. But I also work later than I used to because the stopping point is less obvious without the physical act of leaving a building. It's a trade-off I'm still figuring out the math on.`,
  },
  {
    label: 'Human — opinion piece (moderate length)',
    text: `There's a particular kind of frustration that comes from watching a sport you used to love turn into something you barely recognize. I've followed basketball my whole life, grew up watching the mid-range game, the post play, the pure craftsmanship of someone who could score from anywhere on the court. What we have now is a different sport. Not worse, necessarily, but different in ways that feel like they've stripped something essential out.

The three-pointer didn't ruin basketball. Analytics didn't ruin basketball. But the optimization of basketball — the relentless, data-driven pursuit of efficiency above all else — has made the game more predictable in a way that drains some of the joy out of watching it. When every team is running basically the same offensive system because the math says it's optimal, you lose the idiosyncratic beauty of teams that played weird and interesting ways because their coach had a particular philosophy or their best player had an unusual skill set.

I still watch. I probably always will. But I find myself caring less about the regular season than I used to, and I don't think that's just age.`,
  },
  {
    label: 'AI — academic paragraph (single paragraph)',
    text: `The relationship between social media usage and adolescent mental health has garnered significant scholarly attention in recent years. Several longitudinal studies have demonstrated correlations between high social media consumption and elevated rates of anxiety and depression among teenagers. However, it is important to note that correlation does not imply causation, and researchers have highlighted the complexity of these relationships. The mechanisms through which social media may influence mental health outcomes remain a subject of ongoing investigation. Social comparison theory provides one useful framework for understanding these dynamics, suggesting that frequent exposure to idealized representations of others' lives may negatively impact self-esteem. Additionally, the displacement of sleep and physical activity by screen time may contribute to poorer mental health outcomes independently of social media content itself.`,
  },
];

// ─── Run and display ───────────────────────────────────────────────────────────
console.log('LEXICAL DENSITY MELODY — calibration test\n');
console.log('Score: HIGH = flat melody = AI-like | LOW = irregular melody = human-like\n');
console.log('─'.repeat(72));

samples.forEach(({ label, text }) => {
  const r = calcDensityMelody(text);
  const words = (text.match(/\b[a-z]+\b/gi) || []).length;
  const bar = '█'.repeat(Math.round(r.score / 5)).padEnd(20);
  console.log(`\n[${label}]`);
  if (r.reason) {
    console.log(`  words=${words}  mean_density=${r.mean}  CV=${r.cv}  → ${r.reason}`);
    console.log(`  ► MELODY SCORE: ${r.score}% (neutral)`);
  } else {
    console.log(`  words=${words}  curve_len=${r.curveLen}  mean_density=${r.mean}  std=${r.std}`);
    console.log(`  CV=${r.cv}  gapCV=${r.gapCV}  hfEnergy=${r.hfEnergy}`);
    console.log(`  sub-scores → cv:${r.cvScore}  gapCV:${r.gapCVScore}  hf:${r.hfScore}`);
    console.log(`  ► MELODY SCORE: ${r.score}%  ${bar}`);
  }
});

console.log('\n' + '─'.repeat(72));
console.log('SUMMARY');
console.log('─'.repeat(72));
samples.forEach(({ label, text }) => {
  const r = calcDensityMelody(text);
  const isAI = label.startsWith('AI');
  const correct = isAI ? r.score >= 55 : r.score < 45;
  const flag = r.reason ? '?' : correct ? '✓' : '✗';
  console.log(`  ${flag}  ${String(r.score).padStart(3)}%  ${isAI ? 'AI   ' : 'HUMAN'}  ${label}`);
});
