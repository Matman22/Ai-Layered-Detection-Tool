/**
 * Manual test harness — runs full L1+L5 detection on formatted text samples.
 * Reflects current index.html logic including the paragraph-fix.
 * Usage: node eval/manual_test.js
 */

// ─── Core utilities ───────────────────────────────────────────────────────────
function tokenize(text) { return text.toLowerCase().match(/\b[a-z']+\b/g) || []; }
function getSentences(text) { return text.match(/[^.!?]+[.!?]+/g) || [text]; }
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
function getNaturalParagraphs(text) { return text.split(/\n\s*\n/).filter(p => p.trim().length > 20); }

// ─── L1 signals ───────────────────────────────────────────────────────────────
const AI_PHRASES = ['in conclusion','it is worth noting','it is important to note','furthermore',
  'moreover','in summary','to summarize','delve into','it is crucial','in the realm of',
  'as we explore','it becomes evident','navigating','multifaceted','nuanced','at its core',
  'it is essential','foster','leveraging','paradigm','tapestry','underscore','pivotal','utilize',
  "in today's world",'in the context of','underpins','embodies','holistic','robust','streamline',
  'synergy','it goes without saying','needless to say','when it comes to','let us','one can argue',
  'it should be noted','in light of','it is clear that','first and foremost','last but not least'];
const HEDGE_WORDS = ['perhaps','possibly','might','may','could','seem','appears','generally',
  'typically','often','usually','tend to','suggest','indicate'];

function calcPerplexity(text) {
  const words = tokenize(text); if (!words.length) return 0;
  const lengths = words.map(w => w.length);
  const avg = lengths.reduce((a,b)=>a+b,0)/lengths.length;
  const variance = lengths.reduce((a,b)=>a+Math.pow(b-avg,2),0)/lengths.length;
  return Math.max(0, Math.min(100, 100-(variance*12)));
}
function calcBurstiness(text) {
  const sentences = getSentences(text); if (sentences.length < 3) return 50;
  const lengths = sentences.map(s => s.trim().split(/\s+/).length);
  const avg = lengths.reduce((a,b)=>a+b,0)/lengths.length;
  const std = Math.sqrt(lengths.reduce((a,b)=>a+Math.pow(b-avg,2),0)/lengths.length);
  return Math.max(0, Math.min(100, 100-(std/avg*90)));
}
function calcLexicalDiversity(text) {
  const words = tokenize(text); if (words.length < 10) return 50;
  return Math.max(0, Math.min(100, 100-(new Set(words).size/words.length*130)));
}
function calcAIPhrases(text) {
  const lower = text.toLowerCase(); let hits = 0;
  AI_PHRASES.forEach(p => { if (lower.includes(p)) hits++; });
  return Math.min(100, hits/(tokenize(text).length/100)*25);
}
function calcHedging(text) {
  const lower = text.toLowerCase(); const words = tokenize(text); let count = 0;
  HEDGE_WORDS.forEach(h => { const m = lower.match(new RegExp('\\b'+h+'\\b','g')); if (m) count += m.length; });
  return Math.min(100, count/(words.length/100)*18);
}
function calcPassiveVoice(text) {
  const sentences = getSentences(text); let passive = 0;
  const re = /\b(is|are|was|were|be|been|being)\s+\w+ed\b/gi;
  sentences.forEach(s => { if (re.test(s)) passive++; re.lastIndex=0; });
  return Math.min(100, passive/sentences.length*160);
}
function calcTransitions(text) {
  const ts = ['however','therefore','thus','consequently','nevertheless','additionally',
    'furthermore','moreover','in contrast','on the other hand','as a result',
    'for example','for instance','in addition','similarly'];
  const lower = text.toLowerCase(); let count = 0;
  ts.forEach(t => { const m = lower.match(new RegExp('\\b'+t+'\\b','g')); if (m) count+=m.length; });
  return Math.min(100, count/(tokenize(text).length/100)*22);
}
function calcClauseDepth(text) {
  const sentences = getSentences(text);
  const re = /\b(which|that|although|because|since|while|when|where|if|unless|whereas|whether)\b/gi;
  let total = 0;
  sentences.forEach(s => { const m = s.match(re); total += m ? m.length : 0; });
  const avg = total/sentences.length;
  return Math.max(0, avg>0.8&&avg<2.5 ? Math.min(100,(2.5-Math.abs(avg-1.6))*50) : 30);
}
function calcPunctuationVariance(text) {
  const sentences = getSentences(text).filter(s => s.trim().length > 10);
  if (sentences.length < 4) return 50;
  const counts = sentences.map(s => (s.match(/[,;:—\-()]/g)||[]).length);
  const avg = counts.reduce((a,b)=>a+b,0) / counts.length;
  if (avg < 0.3) return 50;
  const variance = counts.reduce((a,b)=>a+Math.pow(b-avg,2),0) / counts.length;
  const cv = Math.sqrt(variance) / avg;
  return Math.max(0, Math.min(100, Math.round(90 - cv * 55)));
}
function calcParagraphUniformity(text) {
  const paras = getNaturalParagraphs(text); if (paras.length < 2) return 50;
  const lengths = paras.map(p => p.trim().split(/\s+/).length);
  const avg = lengths.reduce((a,b)=>a+b,0)/lengths.length;
  const cv = Math.sqrt(lengths.reduce((a,b)=>a+Math.pow(b-avg,2),0)/lengths.length)/avg;
  return Math.max(0, Math.min(100, 90-(cv*80)));
}
function calcRareWords(text) {
  const common = new Set(['the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','up','about','into','through','during','is','are','was','were','be','been','have','has','had','do','does','did','will','would','could','should','may','might','can','this','that','these','those','it','its','i','you','he','she','we','they','my','your','his','her','our','their','what','which','who','when','where','how','if','then','than','so','as','not','no','there','here','more','also','just','only','very','much','many','most','some','all','any','each','other','over']);
  const words = tokenize(text); const content = words.filter(w => !common.has(w) && w.length > 3);
  if (!content.length) return 50;
  return Math.max(0, Math.min(100, 85-(new Set(content).size/content.length*90)));
}
function calcFormalityShift(text) {
  const sentences = getSentences(text); if (sentences.length < 4) return 50;
  const inf = /\b(i'm|it's|don't|can't|won't|isn't|you're|we're|they're|gonna|wanna|kinda|sorta|yeah|ok|okay|alright|stuff|things|get|got|pretty|really|actually|basically|literally|honestly|like)\b/gi;
  const frm = /\b(therefore|subsequently|notwithstanding|whilst|regarding|pertaining|aforementioned|henceforth|hereby|therein|pursuant|whereby)\b/gi;
  const infScores = sentences.map(s => (s.match(inf)||[]).length);
  const totalInformal = infScores.reduce((a,b)=>a+b,0);
  const avg = infScores.reduce((a,b)=>a+b,0)/infScores.length;
  const std = Math.sqrt(infScores.reduce((a,b)=>a+Math.pow(b-avg,2),0)/infScores.length);
  let score = Math.max(0, Math.min(100, 75-(std*30)));
  if (totalInformal === 0 && sentences.length > 5) score = Math.min(score+20, 100);
  return score;
}
function calcNgramRepetition(text) {
  const words = tokenize(text); if (words.length < 30) return 50;
  const trigrams = {}; for (let i = 0; i < words.length-2; i++) { const k = words[i]+' '+words[i+1]+' '+words[i+2]; trigrams[k]=(trigrams[k]||0)+1; }
  const repeated = Object.values(trigrams).filter(v=>v>1).length;
  return Math.min(100, repeated/Object.keys(trigrams).length*300);
}
function calcSentenceOpenerDiversity(text) {
  const sentences = getSentences(text); if (sentences.length < 5) return 50;
  const openers = sentences.map(s => { const w = s.trim().toLowerCase().match(/\b[a-z']+\b/g)||[]; return w.slice(0,2).join(' '); }).filter(o=>o.length>0);
  return Math.max(0, Math.min(100, 90-(new Set(openers).size/openers.length*95)));
}
function calcPunctuationFingerprint(text) {
  const sentences = getSentences(text); const total = sentences.length; if (total < 4) return 50;
  const emDash = (text.match(/—/g)||[]).length/total;
  const paren = sentences.filter(s=>/\(.*?\)/.test(s)).length/total;
  const aiDash = (text.match(/\w\s*—\s*\w/g)||[]).length/total;
  let score = 0;
  if (emDash>0.4) score+=35; else if (emDash>0.2) score+=15;
  if (aiDash>0.3) score+=25;
  if (paren>0.25&&paren<0.5) score+=20;
  if ((text.match(/!/g)||[]).length===0&&total>8) score+=10;
  return Math.min(100, score);
}
function calcVocabClustering(text) {
  const paragraphs = getParagraphs(text); if (paragraphs.length < 2) return 50;
  const allWords = tokenize(text); const freq = {};
  allWords.forEach(w => { if (w.length>4) freq[w]=(freq[w]||0)+1; });
  const keyTerms = new Set(Object.keys(freq).filter(w=>freq[w]>=2));
  if (keyTerms.size===0) return 30;
  const densities = paragraphs.map(p => { const pw = tokenize(p); return pw.filter(w=>keyTerms.has(w)).length/pw.length; });
  const avg = densities.reduce((a,b)=>a+b,0)/densities.length;
  const cv = avg>0 ? Math.sqrt(densities.reduce((a,b)=>a+Math.pow(b-avg,2),0)/densities.length)/avg : 0;
  return Math.max(0, Math.min(100, 85-(cv*75)));
}

// 17. Density Melody — flat information-weight curve across the doc = AI signal
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
  if (l <= 3) return 0.15;
  if (l <= 6) return 0.40;
  if (l <= 9) return 0.72;
  return 1.0;
}
function calcDensityMelody(text) {
  const words = text.toLowerCase().match(/\b[a-z]+\b/g) || [];
  if (words.length < 80) return 50;
  const wts = words.map(densityWordWeight);
  const W = 25;
  if (words.length < W * 2) return 50;
  const curve = [];
  for (let i = 0; i <= wts.length - W; i++) {
    let s = 0; for (let k = 0; k < W; k++) s += wts[i+k]; curve.push(s/W);
  }
  const n = curve.length;
  const mean = curve.reduce((a,b)=>a+b,0)/n;
  const variance = curve.reduce((a,b)=>a+(b-mean)**2,0)/n;
  const cv = mean > 0.01 ? Math.sqrt(variance)/mean : 0;
  if (mean < 0.32) return 50; // casual text — signal not applicable
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
  return Math.round(cvScore*0.45+gapCVScore*0.30+hfScore*0.25);
}

// ─── L5 signals ───────────────────────────────────────────────────────────────
function calcContractionConsistency(text) {
  const paragraphs = getParagraphs(text); if (paragraphs.length < 3) return 50;
  const re = /\b(i'm|it's|don't|can't|won't|isn't|aren't|wasn't|weren't|you're|we're|they're|i've|you've|we've|they've|i'd|you'd|he'd|she'd|we'd|they'd|i'll|you'll|he'll|she'll|we'll|they'll|hasn't|haven't|hadn't|doesn't|didn't|wouldn't|couldn't|shouldn't|that's|who's|what's|there's|here's)\b/gi;
  const rates = paragraphs.map(p => { const w = tokenize(p).length||1; return (p.match(re)||[]).length/w; });
  const avg = rates.reduce((a,b)=>a+b,0)/rates.length;
  if (avg < 0.005) return 30;
  const cv = Math.sqrt(rates.reduce((a,b)=>a+Math.pow(b-avg,2),0)/rates.length)/avg;
  return Math.max(0, Math.min(100, 88-(cv*75)));
}
function calcOxfordCommaConsistency(text) {
  const with_ = (text.match(/\b\w+\b,\s+\b\w+\b,\s+and\s+\b\w+\b/gi)||[]).length;
  const without = Math.max(0,(text.match(/\b\w+\b,\s+\b\w+\b\s+and\s+\b\w+\b/gi)||[]).length-with_);
  const total = with_+without; if (total < 2) return 50;
  return Math.max(0, Math.min(100, 90-(Math.min(with_,without)/total*180)));
}
function calcNumberFormattingConsistency(text) {
  const w = (text.match(/\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b/gi)||[]).length;
  const d = (text.match(/\b([1-9]|1[0-2])\b/g)||[]).length;
  const total = w+d; if (total < 3) return 50;
  return Math.max(0, Math.min(100, 85-(Math.min(w,d)/total*150)));
}
function calcPrepositionalEndingConsistency(text) {
  const paragraphs = getParagraphs(text); if (paragraphs.length < 3) return 50;
  const re = /\b(in|on|at|to|for|of|with|by|from|up|about|into|out|off|over|under|after|before)\s*[.!?]\s*$/i;
  const rates = paragraphs.map(p => { const s = getSentences(p); return s.filter(x=>re.test(x.trim())).length/(s.length||1); });
  const avg = rates.reduce((a,b)=>a+b,0)/rates.length;
  const std = Math.sqrt(rates.reduce((a,b)=>a+Math.pow(b-avg,2),0)/rates.length);
  return Math.max(0, Math.min(100, 85-(std*400)));
}
function calcParagraphOpenerConsistency(text) {
  const paragraphs = getParagraphs(text); if (paragraphs.length < 3) return 50;
  const articles = new Set(['the','a','an']);
  const pronouns = new Set(['i','you','he','she','it','we','they','this','these','those','my','your','his','her','its','our','their']);
  const transitions = new Set(['however','therefore','thus','moreover','furthermore','additionally','nevertheless','consequently','first','second','third','finally','lastly','overall','ultimately','in','for','but','yet','while','although','since','because','despite']);
  const classify = w => articles.has(w)?'article':pronouns.has(w)?'pronoun':transitions.has(w)?'transition':'other';
  const classes = paragraphs.map(p => { const w = p.trim().toLowerCase().match(/\b[a-z]+\b/g)||[]; return w.length?classify(w[0]):'other'; });
  const freq = {}; classes.forEach(c => { freq[c]=(freq[c]||0)+1; });
  const dominant = Math.max(...Object.values(freq));
  return Math.max(0, Math.min(100, (dominant/classes.length-0.45)/0.55*100));
}

// ─── Scoring ──────────────────────────────────────────────────────────────────
const WEIGHTS = [0.01,0.12,0.08,0.07,0.01,0.01,0.06,0.02,0.10,0.10,0.02,0.10,0.05,0.05,0.07,0.06,0.07];
const NAMES   = ['Perplexity','Burstiness','Lexical diversity','AI phrases','Hedging','Passive voice',
  'Transitions','Clause depth','Punct variance','Para uniformity','Rare words','Register stability',
  'N-gram repeat','Opener diversity','Punct fingerprint','Vocab clustering','Density melody'];

function runDetection(text) {
  const wordCount = tokenize(text).length;
  const reliability = wordCount<100?0.6:wordCount<300?0.82:wordCount<600?0.92:1.0;
  const raw = [calcPerplexity,calcBurstiness,calcLexicalDiversity,calcAIPhrases,calcHedging,
    calcPassiveVoice,calcTransitions,calcClauseDepth,calcPunctuationVariance,calcParagraphUniformity,
    calcRareWords,calcFormalityShift,calcNgramRepetition,calcSentenceOpenerDiversity,
    calcPunctuationFingerprint,calcVocabClustering,calcDensityMelody].map(fn => {
      const result = fn(text); return typeof result === 'object' ? result.score : result;
    });
  // Evidence accumulation: only signals above neutral (50) contribute.
  // Absence of an AI pattern is neutral, not human evidence.
  const evidence = raw.reduce((sum, s, i) => sum + Math.max(0, s - 50) * WEIGHTS[i], 0);
  const baseComposite = Math.min(100, Math.round(evidence * 5));

  // Non-linear triggers: convergence bonus when anchor signals cluster together.
  // Anchor signals are the 4 most AI-specific, reliable indicators.
  // indices: 1=burstiness, 3=AI phrases, 11=register stability, 15=vocab clustering
  const ANCHOR_IDX = [1, 3, 11, 15];
  const anchorsHot = ANCHOR_IDX.filter(i => raw[i] > 70).length;
  const CONV_BONUS = [0, 0, 5, 12, 18]; // bonus for 0/1/2/3/4 anchors above threshold
  // Smoking gun: dense AI phrase usage is a near-certain signal
  const smokingGun = raw[3] > 85 ? 8 : 0;
  const composite = Math.min(100, baseComposite + CONV_BONUS[Math.min(anchorsHot, 4)] + smokingGun);

  const adjusted = composite * reliability + 25 * (1 - reliability);

  const contraction = calcContractionConsistency(text);
  const oxford      = calcOxfordCommaConsistency(text);
  const numbers     = calcNumberFormattingConsistency(text);
  const prepEnding  = calcPrepositionalEndingConsistency(text);
  const openers     = calcParagraphOpenerConsistency(text);
  const l5 = Math.round(contraction*0.25+oxford*0.20+numbers*0.20+prepEnding*0.15+openers*0.20);

  const combined = Math.round(Math.min(100, adjusted*0.75 + 0*0.15 + l5*0.10));
  const paras = getParagraphs(text).length;
  const sents = getSentences(text).length;

  return { raw, baseComposite, composite, adjusted: Math.round(adjusted), l5, combined, reliability, wordCount, sents, paras,
           anchorsHot, convBonus: CONV_BONUS[Math.min(anchorsHot, 4)], smokingGun,
           l5signals: { contraction, oxford, numbers, prepEnding, openers } };
}

function verdict(score) {
  if (score >= 50) return 'LIKELY AI';
  if (score >= 20) return 'AMBIGUOUS';
  return 'LIKELY HUMAN';
}

function printResult(label, text) {
  const r = runDetection(text);
  const bar = score => '█'.repeat(Math.round(score/5)).padEnd(20);
  console.log(`\n${'─'.repeat(62)}`);
  console.log(`[${label}]`);
  console.log(`words=${r.wordCount}  sentences=${r.sents}  paragraphs=${r.paras} (virtual=${getNaturalParagraphs(text).length<3?'yes':'no'})  reliability=${r.reliability}`);
  console.log(`${'─'.repeat(62)}`);
  r.raw.forEach((s, i) => {
    const bar = '█'.repeat(Math.round(s/5)).padEnd(20);
    const flag = s >= 65 ? ' ◄' : '';
    console.log(`  ${NAMES[i].padEnd(20)} ${String(Math.round(s)).padStart(3)}%  ${bar}${flag}`);
  });
  console.log(`  ${'─'.repeat(58)}`);
  console.log(`  ${'L1 base'.padEnd(20)} ${r.baseComposite}%`);
  const bonusStr = r.convBonus || r.smokingGun ? ` (+${r.convBonus} conv[${r.anchorsHot} anchors]${r.smokingGun ? ` +${r.smokingGun} 🔫` : ''})` : '';
  console.log(`  ${'L1 composite'.padEnd(20)} ${r.composite}%${bonusStr}`);
  console.log(`  ${'L5 consistency'.padEnd(20)} ${r.l5}%  (contr:${r.l5signals.contraction} oxf:${r.l5signals.oxford} num:${r.l5signals.numbers} prep:${r.l5signals.prepEnding} open:${r.l5signals.openers})`);
  console.log(`\n  ► FINAL SCORE: ${r.combined}%  →  ${verdict(r.combined)}`);
}

// ─── Test samples ─────────────────────────────────────────────────────────────

const samples = [

  { label: 'AI — formal essay (GPT-style, single paragraph)',
    text: `Artificial intelligence has fundamentally transformed the landscape of modern technology. It is worth noting that the implications of this transformation extend far beyond the realm of computer science. Furthermore, the integration of AI systems into everyday workflows has raised important questions about the future of human labor. It is important to note that these concerns are not without merit. Moreover, the ethical dimensions of AI deployment must be carefully considered by policymakers and technology leaders alike. In conclusion, the responsible development of artificial intelligence requires a nuanced understanding of its multifaceted impact on society. Consequently, stakeholders must engage in meaningful dialogue to address these pivotal challenges. The landscape of AI governance is evolving rapidly, and it is essential to foster collaboration between diverse perspectives. Holistic approaches that consider both technical and social dimensions will ultimately determine the success of AI integration in modern society. The paradigm shift we are witnessing underscores the need for robust regulatory frameworks that can adapt to rapid technological change.` },

  { label: 'Human — personal essay (casual, single paragraph)',
    text: `I never thought I'd be the kind of person who cares about gardening. Honestly, it started as a joke — my roommate bet me twenty dollars I couldn't keep a cactus alive for three months. I lost that bet spectacularly. But somewhere between killing that first cactus and my fourth failed attempt at growing basil, something clicked. There's something weirdly therapeutic about soil under your fingernails. My mom always had a garden when I was growing up, and I remember thinking it was the most boring thing imaginable. Now I get it. You're not really growing plants, you're just slowing down. The tomatoes I finally got right last summer tasted completely different from anything from a grocery store. Not necessarily better — just more real somehow. My neighbor Sarah actually stopped to ask me about them, which honestly made my whole month.` },

  { label: 'AI — multi-paragraph essay (with real breaks)',
    text: `Remote work has fundamentally transformed the way organizations operate in the modern economy. It is worth noting that this transformation has accelerated significantly in recent years, particularly following the global pandemic that began in 2020. Furthermore, the implications of this shift extend far beyond simple logistical changes to office arrangements.

It is important to note that remote work offers numerous advantages for both employees and employers. Workers benefit from the elimination of lengthy commutes, greater flexibility in managing personal responsibilities, and the ability to create customized work environments. Employers, meanwhile, can access a broader talent pool unrestricted by geographic limitations.

However, it is crucial to acknowledge that remote work also presents significant challenges that organizations must carefully navigate. Communication barriers can emerge when teams are distributed across different time zones. Moreover, the boundaries between professional and personal life can become blurred, potentially leading to overwork and burnout.

Consequently, organizations that successfully implement remote work policies tend to adopt structured approaches that address these challenges proactively. Regular virtual meetings, clear communication protocols, and deliberate efforts to maintain team cohesion are essential components. In conclusion, remote work represents a paradigm shift in organizational culture that carries both significant opportunities and notable challenges.` },

  { label: 'Human — blog post (casual, multi-paragraph)',
    text: `So I finally tried that new ramen place downtown everyone keeps talking about. Went on a Tuesday thinking it'd be quiet — nope, still a 40-minute wait. Worth it though? Kind of.

The broth was genuinely great, really deep and rich, the kind you can tell took forever. But my noodles were slightly overcooked and the chashu was a bit dry. My friend got the spicy miso and she was raving about it the whole walk home, so maybe I just ordered wrong.

The space itself is tiny and the tables are practically on top of each other — overheard at least three conversations I did not ask to be part of. The couple next to us were clearly on a first date and it was going badly. I felt things.

Would I go back? Probably. Would I wait 40 minutes again? Ask me when I'm not cold and hungry. The prices were reasonable for what you get, and honestly the gyoza were perfect — crispy on the bottom, not greasy at all. That alone might bring me back.` },

  { label: 'AI — academic paragraph (single paragraph)',
    text: `The relationship between social media usage and adolescent mental health has garnered significant scholarly attention in recent years. Several longitudinal studies have demonstrated correlations between high social media consumption and elevated rates of anxiety and depression among teenagers. However, it is important to note that correlation does not imply causation, and researchers have highlighted the complexity of these relationships. The mechanisms through which social media may influence mental health outcomes remain a subject of ongoing investigation. Social comparison theory provides one useful framework for understanding these dynamics, suggesting that frequent exposure to idealized representations of others' lives may negatively impact self-esteem. Additionally, the displacement of sleep and physical activity by screen time may contribute to poorer mental health outcomes independently of social media content itself. It is essential to consider these multifaceted factors when developing evidence-based interventions for adolescent mental health.` },

  { label: 'AI — long formal essay (600+ words, multi-paragraph)',
    text: `Artificial intelligence has emerged as one of the most transformative technologies of the twenty-first century, reshaping industries, redefining labor markets, and challenging longstanding assumptions about human cognition. It is worth noting that the pace of this transformation has accelerated considerably in recent years, driven by advances in deep learning, increased computational power, and the availability of vast datasets. Furthermore, the implications of these developments extend far beyond the realm of computer science, touching virtually every sector of the global economy.

The integration of artificial intelligence into the workforce represents one of the most significant economic shifts of our time. It is important to note that this transition is not without precedent; previous technological revolutions, from industrialization to the advent of digital computing, have similarly displaced certain categories of labor while simultaneously creating new forms of employment. However, it is crucial to acknowledge that the speed and scope of AI-driven automation present unique challenges that policymakers and business leaders must carefully navigate. Consequently, educational institutions and training programs must adapt rapidly to equip workers with the skills necessary to thrive in an AI-augmented economy.

Moreover, the ethical dimensions of artificial intelligence deployment demand rigorous examination. The development of algorithmic systems capable of making consequential decisions in domains such as healthcare, criminal justice, and financial services raises profound questions about accountability, transparency, and fairness. It is essential to foster robust governance frameworks that ensure AI systems are developed and deployed in ways that respect human dignity and promote equitable outcomes. Stakeholders across government, industry, and civil society must engage in meaningful dialogue to address these pivotal concerns before the technology becomes further entrenched in critical infrastructure.

The geopolitical dimensions of artificial intelligence competition also merit careful consideration. In the context of an increasingly multipolar world, leading nations are investing heavily in AI research and development as a strategic priority. It is clear that the nation or coalition that achieves sustained leadership in artificial intelligence capabilities will likely enjoy significant advantages in economic productivity, military capability, and diplomatic influence. This dynamic has prompted calls for international cooperation to establish norms and standards governing the development and deployment of AI systems, particularly in sensitive domains such as autonomous weapons and surveillance technology.

Looking forward, the trajectory of artificial intelligence development will be shaped by a complex interplay of technical, economic, social, and political factors. It is essential to approach this challenge with a nuanced understanding of both the opportunities and risks that advanced AI systems present. Holistic frameworks that consider not only technical performance metrics but also broader societal impacts will be necessary to guide responsible innovation. The paradigm shift we are witnessing underscores the importance of maintaining a long-term perspective, prioritizing human wellbeing alongside technological progress, and ensuring that the benefits of artificial intelligence are distributed equitably across society. In conclusion, the path forward requires sustained collaboration among diverse stakeholders committed to leveraging AI's transformative potential while safeguarding the values and institutions that underpin democratic societies.` },

  { label: 'Human — long personal essay (600+ words, multi-paragraph)',
    text: `My grandmother kept every letter anyone ever sent her in a shoebox under her bed. When she died we found forty years of correspondence in there — birthday cards from cousins she'd lost touch with, a handful of letters from my grandfather written during the two years he worked on a fishing boat in Alaska before they got married, notes from neighbors about borrowed things, a postcard from someone named Vivienne postmarked 1978 with no explanation. My grandmother was not a sentimental person in any obvious way. She didn't cry at movies. She gave practical gifts. But she kept every piece of paper anyone had ever handed her, and I've been thinking about that shoebox a lot lately.

I'm the kind of person who deletes emails after reading them. Not out of any organizational philosophy — I'm not organized — but because the accumulation makes me anxious. Thousands of messages sitting there unread feels like debt. So I delete, and I don't feel bad about it, and then occasionally I want to find something and can't, and I feel briefly bad, and then I forget about it. My digital life is full of these small erasures.

The thing is, I don't actually know what I've lost. That's the nature of deletion: you can't audit the archive you didn't keep. My grandmother could go back through that shoebox and tell you exactly who had been in her life and when and how. There's a card in there from a woman named Patricia that reads only "Thank you for everything, you'll never know how much." My grandmother is gone now and I'll never know what that meant, but I know Patricia existed and that something happened between them that mattered.

I spent an afternoon with those letters after the funeral, reading through them with my aunt. What struck me most wasn't the content exactly but the texture of the handwriting — how you could see my grandfather getting tired toward the end of a long letter, or how my grandmother's friend Carol always drew a small star at the end of her sentences instead of periods. Nobody does that in email. There's no equivalent of the slightly crooked stamp or the coffee ring on the envelope. Digital communication is clean in a way that real life isn't.

I know there's a version of this essay where I conclude that we should all write more letters and preserve more things and be more intentional about memory. But I don't think that's quite right either. My grandmother wasn't trying to be intentional. She just didn't throw things away. The archive was accidental, which might be why it feels so honest.

What I think I'm actually mourning when I think about the shoebox isn't the letters themselves but the evidence they provide that people move through your life and leave marks. Patricia thanked my grandmother for something my grandmother apparently never told anyone about. That exists somewhere in the record now. Most of what I do doesn't exist anywhere. I send a message, the other person reads it, we have an exchange, I delete it, they probably delete it, and it's as though it never happened — except that it did, except that we are slightly different people because it happened, except that the mark is there even if the paper isn't.

That's probably fine. I'm not going to start keeping shoeboxes. But I do think about Vivienne and her unexplained postcard from 1978, and I wonder what I'm not leaving behind.` },

  { label: 'Human — opinion piece (moderate length)',
    text: `There's a particular kind of exhaustion that comes from living somewhere you used to love. I've been in Philadelphia for six years now, and somewhere around year four I stopped noticing the things that made me move here in the first place — the weird little bars, the fact that you can actually afford an apartment with a second bedroom, the way the city kind of just lets you be. Now I mostly notice the potholes and the fact that my gym raised prices again.

I don't think this is a Philadelphia problem. I think it's a me problem, or maybe just a humans-in-general problem. We habituate. The good stuff fades into background and the annoyances stay sharp because annoyances require action. You can't just ignore a pothole the same way you stop seeing a mural.

What I've been trying is embarrassingly simple: I've started walking different routes. Not for exercise, just to see different blocks. Last week I ended up on a street I'd somehow never been on despite living eight blocks away. There was a guy repairing clocks in a window. I don't know why that hit me the way it did, but it kind of reset something.` },
];

// ─── Run all ──────────────────────────────────────────────────────────────────
console.log('AI ORIGIN DETECTOR — MANUAL TEST RESULTS');
console.log('Updated detection: paragraph-fix active\n');
samples.forEach(({ label, text }) => printResult(label, text));

console.log(`\n${'─'.repeat(62)}`);
console.log('SUMMARY');
console.log('─'.repeat(62));
samples.forEach(({ label, text }) => {
  const r = runDetection(text);
  const expected = label.startsWith('AI') ? 'AI' : 'Human';
  const got = verdict(r.combined);
  const correct = (expected === 'AI' && got !== 'LIKELY HUMAN') || (expected === 'Human' && got !== 'LIKELY AI');
  console.log(`  ${correct?'✓':'✗'} ${String(r.combined+'%').padStart(4)}  ${got.padEnd(14)}  ${label}`);
});
