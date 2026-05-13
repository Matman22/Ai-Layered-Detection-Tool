/**
 * AI Phrases signal recalibration.
 *
 * Problem with current formula: hits/words * 1800
 *   - 3 clear AI phrases in 186 words = 29% (below neutral — contributes nothing)
 *   - Requires ~6 phrases per 186 words to reach 50%
 *   - Treats all phrases equally regardless of how AI-specific they are
 *   - Doesn't reward finding multiple distinct phrases together
 *
 * New design:
 *   Tier 1 — near-certain LLM fingerprints: even 1 hit → above neutral (55+%)
 *   Tier 2 — common AI filler/transition language: need 2+ hits to contribute
 *   Non-linear curve with combo bonus for co-occurrence
 *
 * Usage: node eval/test_ai_phrases_recal.js
 */

// ─── Current formula (baseline) ───────────────────────────────────────────────
const OLD_PHRASES = ['in conclusion','it is worth noting','it is important to note','furthermore',
  'moreover','in summary','to summarize','delve into','it is crucial','in the realm of',
  'as we explore','it becomes evident','navigating','multifaceted','nuanced','at its core',
  'it is essential','foster','leveraging','paradigm','tapestry','underscore','pivotal','utilize',
  "in today's world",'in the context of','underpins','embodies','holistic','robust','streamline',
  'synergy','it goes without saying','needless to say','when it comes to','let us','one can argue',
  'it should be noted','in light of','it is clear that','first and foremost','last but not least'];

function calcAIPhrasesOld(text) {
  const lower = text.toLowerCase();
  const wc = text.toLowerCase().match(/\b[a-z']+\b/g).length || 1;
  const hits = OLD_PHRASES.filter(p => lower.includes(p)).length;
  return Math.min(100, Math.round(hits / wc * 1800));
}

// ─── New tiered formula ────────────────────────────────────────────────────────
// Tier 1: phrases that are near-exclusive to LLM output.
// A single occurrence is meaningful evidence.
const TIER1 = [
  'it is worth noting',      // very rare in genuine human writing
  'it is important to note',
  "it's worth noting",
  'it should be noted',
  'it is important to understand',
  'it is clear that',
  'it becomes evident',
  'it goes without saying',
  'needless to say',
  'delve into',              // nearly always AI
  'delves into',
  'multifaceted',
  'at its core',
  'in the realm of',
  'as we explore',
  'tapestry',                // AI loves this word
  'underscore',              // as a verb ("underscore the importance")
  'pivotal',
  'embodies',
  'underpins',
  'first and foremost',
  'last but not least',
  "in today's world",
  "in today's fast-paced",
  "in today's ever-changing",
  'it is crucial to note',
  'it is essential to note',
  'one can argue',
];

// Tier 2: phrases common in AI output, but also appear in careful human writing.
// Multiple hits together are the signal.
const TIER2 = [
  'in conclusion',
  'furthermore',
  'moreover',
  'in summary',
  'to summarize',
  'it is crucial',
  'it is essential',
  'in the context of',
  'leveraging',
  'foster',
  'paradigm',
  'robust',
  'streamline',
  'synergy',
  'utilize',
  'navigating',
  'holistic',
  'nuanced',
  'in light of',
  'when it comes to',
  'let us',
  'consequently',
  'notably',
  'in addition',
  'moving forward',
  'going forward',
  'at the end of the day',
  'best practices',
  'leverage',               // as standalone verb
  'game-changer',
  'cutting-edge',
  'proactive',
  'actionable',
  'impactful',
  'transformative',
  'stakeholders',
];

function calcAIPhrasesNew(text) {
  const lower = text.toLowerCase();

  const tier1Hits = TIER1.filter(p => lower.includes(p)).length;
  const tier2Hits = TIER2.filter(p => lower.includes(p)).length;

  // Tier 1 scoring: first hit is a strong signal, diminishing returns after
  // 1 hit → 58 (above neutral), 2 → 70, 3 → 80, 4+ → 88+
  const t1Score = tier1Hits === 0 ? 0
    : Math.min(95, 58 + (tier1Hits - 1) * 12);

  // Tier 2 scoring: needs multiple hits before it means much
  // 1 hit → 15, 2 → 27, 3 → 37, 4+ → 44+
  const t2Score = tier2Hits === 0 ? 0
    : tier2Hits === 1 ? 15
    : Math.min(55, 27 + (tier2Hits - 2) * 8);

  // Combo bonus: finding both tier types together is a stronger combined signal
  const comboBonus = tier1Hits >= 1 && tier2Hits >= 2 ? 12
    : tier1Hits >= 1 && tier2Hits >= 1 ? 5
    : 0;

  return Math.min(100, t1Score + t2Score + comboBonus);
}

// ─── Test samples ──────────────────────────────────────────────────────────────
const samples = [
  // ── Should score HIGH ──────────────────────────────────────────────────────
  { label: 'AI — formal essay (heavy AI phrases)', expected: 'high',
    text: `Artificial intelligence has fundamentally transformed the landscape of modern technology. It is worth noting that the implications of this transformation extend far beyond the realm of computer science. Furthermore, the integration of AI systems into everyday workflows has raised important questions about the future of human labor. It is important to note that these concerns are not without merit. Moreover, the ethical dimensions of AI deployment must be carefully considered by policymakers and technology leaders alike. In conclusion, the responsible development of artificial intelligence requires a nuanced understanding of its multifaceted impact on society. Consequently, stakeholders must engage in meaningful dialogue to address these pivotal challenges.` },

  { label: 'AI — listicle (3 tier-1 phrases, diluted by length)', expected: 'high',
    text: `In today's fast-paced world, managing your time effectively has become more important than ever. Whether you are a student, a professional, or an entrepreneur, developing strong time management skills can significantly impact your productivity and overall wellbeing. It is worth noting that many high-achieving individuals attribute their success not to working harder, but to working smarter. Furthermore, research consistently demonstrates that those who implement structured time management strategies report lower stress levels and higher job satisfaction. One of the most effective approaches involves breaking your day into focused work blocks. Additionally, prioritizing tasks using frameworks such as the Eisenhower Matrix can help you distinguish between what is urgent and what is truly important. It is important to note that digital distractions represent one of the greatest threats to sustained focus in the modern workplace. Consequently, establishing clear boundaries around technology use during dedicated work periods is essential.` },

  { label: 'AI — email draft (professional GPT language)', expected: 'high',
    text: `I hope this message finds you well. I am writing to follow up on our discussion from last week regarding the proposed timeline for the Henderson account deliverables. After careful consideration and consultation with the relevant stakeholders, I wanted to provide an update on where things currently stand. It is important to note that the team has made significant progress on the initial phase of the project, and we are on track to meet the preliminary milestones outlined in our initial planning document. Furthermore, we have identified several opportunities to streamline the workflow that I believe will contribute meaningfully to our overall efficiency. Moving forward, I would like to schedule a brief call at your earliest convenience to discuss these developments in greater detail.` },

  { label: 'AI — academic abstract (sparse phrases)', expected: 'medium',
    text: `This study investigates the relationship between urban green space accessibility and psychological wellbeing outcomes among adult residents of metropolitan areas. Using a mixed-methods approach combining spatial analysis of park distribution data with survey-based measures of self-reported stress, anxiety, and life satisfaction, we examine whether proximity to green space is associated with improved mental health indicators after controlling for relevant socioeconomic covariates. Preliminary findings suggest a statistically significant negative relationship between green space proximity and self-reported stress levels. Furthermore, qualitative interview data indicate that the perceived quality and safety of green spaces may moderate the relationship between mere proximity and actual usage rates. These findings have implications for urban planning policy and suggest that equitable distribution of high-quality green infrastructure should be considered a public health intervention.` },

  // ── Should score LOW ───────────────────────────────────────────────────────
  { label: 'HUMAN — personal essay (casual)', expected: 'low',
    text: `I never thought I'd be the kind of person who cares about gardening. Honestly, it started as a joke — my roommate bet me twenty dollars I couldn't keep a cactus alive for three months. I lost that bet spectacularly. But somewhere between killing that first cactus and my fourth failed attempt at growing basil, something clicked. There's something weirdly therapeutic about soil under your fingernails. My mom always had a garden when I was growing up, and I remember thinking it was the most boring thing imaginable. Now I get it. You're not really growing plants, you're just... slowing down. The tomatoes I finally got right last summer tasted completely different from anything from a grocery store.` },

  { label: 'HUMAN — journalism (formal, no AI phrases)', expected: 'low',
    text: `For decades, the conventional wisdom held that raising the minimum wage would reduce employment — that companies, faced with higher labor costs, would simply hire fewer workers. The argument was intuitive and had the backing of introductory economics textbooks. Then David Card and Alan Krueger published a study in 1994 comparing fast-food employment in New Jersey and Pennsylvania after New Jersey raised its minimum wage. Employment, they found, had actually gone up slightly in New Jersey relative to Pennsylvania. The backlash was fierce. Critics questioned the methodology, the data, the framing. Card himself was essentially pushed out of labor economics for years. He won the Nobel Prize in 2021. What happened in the intervening decades was not just a vindication of one study but a wholesale revision of how economists think about labor markets.` },

  { label: 'HUMAN — academic paper (formal human writing)', expected: 'low',
    text: `The phenomenon of code-switching has been studied extensively within sociolinguistics, yet its cognitive underpinnings remain incompletely understood. Researchers have proposed several theoretical frameworks to explain why bilingual speakers alternate between languages within a single discourse context. The Matrix Language Frame model, developed by Carol Myers-Scotton, posits that one language serves as the structural backbone of an utterance while the other contributes lexical items. This model has been applied productively to data from numerous language pairs, though critics have noted its difficulty in accounting for instances of symmetric bilingualism where neither language clearly dominates. More recent approaches draw on dynamic systems theory, treating language choice as an emergent property of complex interactions between the speaker, interlocutor, and communicative context.` },

  { label: 'HUMAN — technical blog (developer)', expected: 'low',
    text: `I spent three days debugging a memory leak that turned out to be six lines of code. Here is what I learned. The app was a Node.js service that processed incoming webhooks and wrote results to a Postgres database. Under load it would gradually eat memory until the process crashed, usually somewhere around the four-hour mark. The first thing I did was add heap snapshots. Nothing useful — the retained objects looked normal. I added more logging. Still nothing obvious. On day two I started reading the ORM source code, which is how I found out that our version of Sequelize had a known issue with connection pool cleanup under certain transaction rollback conditions. The fix was embarrassingly simple — wrap the transaction initialization in a try-catch and explicitly release the connection on failure.` },

  { label: 'HUMAN — opinion piece (argumentative)', expected: 'low',
    text: `There's a particular kind of frustration that comes from watching a sport you used to love turn into something you barely recognize. I've followed basketball my whole life, grew up watching the mid-range game, the post play, the pure craftsmanship of someone who could score from anywhere on the court. What we have now is a different sport. Not worse, necessarily, but different in ways that feel like they've stripped something essential out. The three-pointer didn't ruin basketball. Analytics didn't ruin basketball. But the optimization of basketball — the relentless, data-driven pursuit of efficiency above all else — has made the game more predictable in a way that drains some of the joy out of watching it.` },
];

// ─── Run comparison ───────────────────────────────────────────────────────────
console.log('AI PHRASES — tiered formula vs current formula\n');
console.log('─'.repeat(72));
console.log(`${'Sample'.padEnd(45)} ${'OLD'.padStart(5)} ${'NEW'.padStart(5)}  ${'Tier1/Tier2 hits'}`);
console.log('─'.repeat(72));

samples.forEach(({ label, expected, text }) => {
  const lower = text.toLowerCase();
  const oldScore = calcAIPhrasesOld(text);
  const newScore = calcAIPhrasesNew(text);
  const t1 = TIER1.filter(p => lower.includes(p));
  const t2 = TIER2.filter(p => lower.includes(p));

  const oldFlag = expected === 'high' ? (oldScore >= 50 ? '✓' : '✗') : expected === 'low' ? (oldScore < 30 ? '✓' : '✗') : '~';
  const newFlag = expected === 'high' ? (newScore >= 50 ? '✓' : '✗') : expected === 'low' ? (newScore < 30 ? '✓' : '✗') : '~';

  console.log(`${label.substring(0,44).padEnd(44)} ${String(oldScore).padStart(4)}%${oldFlag} ${String(newScore).padStart(4)}%${newFlag}  T1:[${t1.join(', ')}] T2:[${t2.slice(0,4).join(', ')}${t2.length>4?'...':''}]`);
});

console.log('\n─'.repeat(72));
console.log('Phrase breakdown for key failing case (AI listicle):');
const listicle = samples[1].text.toLowerCase();
console.log('  Tier 1 hits:', TIER1.filter(p => listicle.includes(p)));
console.log('  Tier 2 hits:', TIER2.filter(p => listicle.includes(p)));
console.log('\nPhrase breakdown for human journalism (should score low):');
const journ = samples[5].text.toLowerCase();
console.log('  Tier 1 hits:', TIER1.filter(p => journ.includes(p)));
console.log('  Tier 2 hits:', TIER2.filter(p => journ.includes(p)));
console.log('\nPhrase breakdown for human academic paper:');
const acad = samples[6].text.toLowerCase();
console.log('  Tier 1 hits:', TIER1.filter(p => acad.includes(p)));
console.log('  Tier 2 hits:', TIER2.filter(p => acad.includes(p)));
