// Per-signal breakdown for Monte Carlo window sampling
// Usage: node eval/_signal_breakdown.js

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

function calcBurstiness(t) {
  const s=getSentences(t); if(s.length<3) return 50;
  const l=s.map(x=>x.trim().split(/\s+/).length);
  const avg=l.reduce((sum,b)=>sum+b,0)/l.length;
  const std=Math.sqrt(l.reduce((sum,b)=>sum+(b-avg)**2,0)/l.length);
  return Math.max(0,Math.min(100,100-(std/avg*90)));
}
function calcLexicalDiversity(t) {
  const w=tokenize(t); if(w.length<10) return 50;
  return Math.max(0,Math.min(100,100-(new Set(w).size/w.length*130)));
}
const T1=['it is worth noting','it is important to note',"it's worth noting",'it should be noted','it becomes evident','it goes without saying','needless to say','delve into','multifaceted','at its core','in the realm of','tapestry','underscore','pivotal','embodies','underpins','first and foremost','last but not least',"in today's world","in today's fast-paced",'one can argue','it is crucial to note','it is essential to note'];
const T2=['in conclusion','furthermore','moreover','in summary','to summarize','it is crucial','it is essential','in the context of','leveraging','foster','paradigm','robust','streamline','synergy','utilize','navigating','holistic','nuanced','in light of','when it comes to','let us','consequently','in addition','moving forward','going forward','best practices','impactful','transformative','stakeholders'];
function calcAIPhrases(t) {
  const l=t.toLowerCase();
  const t1=T1.filter(p=>l.includes(p)).length, t2=T2.filter(p=>l.includes(p)).length;
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

const SAMPLES = 20;

const texts = [
  { label: 'AI — long formal essay', text: `Artificial intelligence has emerged as one of the most transformative technologies of the twenty-first century, reshaping industries, redefining labor markets, and challenging longstanding assumptions about human cognition. It is worth noting that the pace of this transformation has accelerated considerably in recent years, driven by advances in deep learning, increased computational power, and the availability of vast datasets. Furthermore, the implications of these developments extend far beyond the realm of computer science, touching virtually every sector of the global economy.

The integration of artificial intelligence into the workforce represents one of the most significant economic shifts of our time. It is important to note that this transition is not without precedent; previous technological revolutions, from industrialization to the advent of digital computing, have similarly displaced certain categories of labor while simultaneously creating new forms of employment. However, it is crucial to acknowledge that the speed and scope of AI-driven automation present unique challenges that policymakers and business leaders must carefully navigate. Consequently, educational institutions and training programs must adapt rapidly to equip workers with the skills necessary to thrive in an AI-augmented economy.

Moreover, the ethical dimensions of artificial intelligence deployment demand rigorous examination. The development of algorithmic systems capable of making consequential decisions in domains such as healthcare, criminal justice, and financial services raises profound questions about accountability, transparency, and fairness. It is essential to foster robust governance frameworks that ensure AI systems are developed and deployed in ways that respect human dignity and promote equitable outcomes. Stakeholders across government, industry, and civil society must engage in meaningful dialogue to address these pivotal concerns before the technology becomes further entrenched in critical infrastructure.

The geopolitical dimensions of artificial intelligence competition also merit careful consideration. In the context of an increasingly multipolar world, leading nations are investing heavily in AI research and development as a strategic priority. It is clear that the nation or coalition that achieves sustained leadership in artificial intelligence capabilities will likely enjoy significant advantages in economic productivity, military capability, and diplomatic influence.

Looking forward, the trajectory of artificial intelligence development will be shaped by a complex interplay of technical, economic, social, and political factors. It is essential to approach this challenge with a nuanced understanding of both the opportunities and risks that advanced AI systems present. Holistic frameworks that consider not only technical performance metrics but also broader societal impacts will be necessary to guide responsible innovation. In conclusion, the path forward requires sustained collaboration among diverse stakeholders committed to leveraging AI transformative potential while safeguarding the values and institutions that underpin democratic societies.` },

  { label: 'HUMAN — personal essay (shoebox)', text: `My grandmother kept every letter anyone ever sent her in a shoebox under her bed. When she died we found forty years of correspondence in there — birthday cards from cousins she had lost touch with, a handful of letters from my grandfather written during the two years he worked on a fishing boat in Alaska before they got married, notes from neighbors about borrowed things, a postcard from someone named Vivienne postmarked 1978 with no explanation. My grandmother was not a sentimental person in any obvious way. She did not cry at movies. She gave practical gifts. But she kept every piece of paper anyone had ever handed her.

I am the kind of person who deletes emails after reading them. Not out of any organizational philosophy — I am not organized — but because the accumulation makes me anxious. Thousands of messages sitting there unread feels like debt. So I delete, and I do not feel bad about it, and then occasionally I want to find something and cannot, and I feel briefly bad, and then I forget about it. My digital life is full of these small erasures.

The thing is, I do not actually know what I have lost. That is the nature of deletion: you cannot audit the archive you did not keep. My grandmother could go back through that shoebox and tell you exactly who had been in her life and when and how. There is a card in there from a woman named Patricia that reads only "Thank you for everything, you will never know how much." My grandmother is gone now and I will never know what that meant, but I know Patricia existed and that something happened between them that mattered.

I spent an afternoon with those letters after the funeral, reading through them with my aunt. What struck me most was the texture of the handwriting — how you could see my grandfather getting tired toward the end of a long letter, or how my grandmother's friend Carol always drew a small star at the end of her sentences instead of periods.

That is probably fine. I am not going to start keeping shoeboxes. But I do think about Vivienne and her unexplained postcard from 1978, and I wonder what I am not leaving behind.` },
];

for (const { label, text } of texts) {
  const words = text.split(/\s+/).filter(w=>w.length>0);
  const wc = words.length;
  const windowPct = wc < 500 ? 0.40 : 0.35;
  const windowSize = Math.max(50, Math.round(wc * windowPct));
  let totB=0,totL=0,totA=0,totF=0,totH=0;
  const perSample = [];
  for (let i=0;i<SAMPLES;i++) {
    const start = Math.floor(Math.random()*(wc-windowSize+1));
    const sub = words.slice(start,start+windowSize).join(' ');
    const b=calcBurstiness(sub), a=calcAIPhrases(sub).score, f=calcFormalityShift(sub), tr=calcTransitions(sub), vc=calcVocabClustering(sub);
    const avg=Math.round((b+a+f+tr+vc)/5);
    totB+=b; totA+=a; totF+=f; totH+=tr; totL+=vc;
    perSample.push(avg);
  }
  const n=SAMPLES;
  const finalAvg=Math.round((totB+totL+totA+totF+totH)/n/5);
  console.log(label+' (wc='+wc+', window='+windowSize+'):');
  console.log('  burstiness:     '+Math.round(totB/n)+'%');
  console.log('  aiPhrases:      '+Math.round(totA/n)+'%');
  console.log('  formalityShift: '+Math.round(totF/n)+'%');
  console.log('  transitions:    '+Math.round(totH/n)+'%');
  console.log('  vocabClustering:'+Math.round(totL/n)+'%');
  console.log('  MEAN SCORE:     '+finalAvg+'%');
  const sScores=perSample;
  const mean=sScores.reduce((a,b)=>a+b,0)/sScores.length;
  const stdDev=Math.sqrt(sScores.reduce((a,b)=>a+(b-mean)**2,0)/sScores.length);
  console.log('  per-sample scores: ['+perSample.join(', ')+']');
  console.log('  stdDev: '+Math.round(stdDev)+'  range: '+(Math.max(...perSample)-Math.min(...perSample)));
  console.log('');
}
