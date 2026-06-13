#!/usr/bin/env python3
"""
raid_test.py — RAID benchmark evaluation for the AI Origin Detector

Ports all detection logic from index.html to Python, downloads a sample of the
RAID ACL-2024 benchmark from HuggingFace, runs every text through the detector,
and prints a detailed accuracy report broken down by model, domain, attack type,
decoding strategy, and repetition penalty.

Usage:
  pip install datasets huggingface_hub
  python raid_test.py [options]

Options:
  --sample N          Rows to sample, stratified by label (default: 2000)
  --split SPLIT       Dataset split: train or test (default: train)
  --no-monte-carlo    Skip Monte Carlo (faster, ~3x speedup)
  --no-density        Skip Density Melody ensemble (faster)
  --domain DOMAIN     Filter to one domain (e.g. news, abstracts, reddit)
  --attack ATTACK     Filter to one attack type (e.g. none, paraphrase)
  --min-words N       Skip texts shorter than N words (default: 50)
  --seed N            Random seed for stratified sampling (default: 42)
  --dataset NAME      HuggingFace dataset name (default: liamdugan/raid)
"""

import re
import math
import random
import argparse
import sys
from collections import defaultdict

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

AI_PHRASES_T1 = [
    'it is worth noting', 'it is important to note', "it's worth noting",
    'it should be noted', 'it is important to understand', 'it is clear that',
    'it becomes evident', 'it goes without saying', 'needless to say',
    'delve into', 'delves into', 'multifaceted', 'at its core', 'in the realm of',
    'as we explore', 'tapestry', 'underscore', 'pivotal', 'embodies', 'underpins',
    'first and foremost', 'last but not least', "in today's world",
    "in today's fast-paced", "in today's ever-changing", 'one can argue',
    'it is crucial to note', 'it is essential to note',
]

AI_PHRASES_T2 = [
    'in conclusion', 'furthermore', 'moreover', 'in summary', 'to summarize',
    'it is crucial', 'it is essential', 'in the context of', 'leveraging', 'foster',
    'paradigm', 'robust', 'streamline', 'synergy', 'utilize', 'navigating',
    'holistic', 'nuanced', 'in light of', 'when it comes to', 'let us',
    'consequently', 'in addition', 'moving forward', 'going forward',
    'best practices', 'game-changer', 'cutting-edge', 'proactive', 'actionable',
    'impactful', 'transformative', 'stakeholders',
]

HEDGE_WORDS = [
    'perhaps', 'possibly', 'might', 'may', 'could', 'seem', 'appears',
    'generally', 'typically', 'often', 'usually', 'tend to', 'suggest', 'indicate',
]

TRANSITION_WORDS = [
    'however', 'therefore', 'thus', 'consequently', 'nevertheless',
    'additionally', 'furthermore', 'moreover', 'in contrast', 'on the other hand',
    'as a result', 'for example', 'for instance', 'in addition', 'similarly',
]

DENSITY_STOPWORDS = set([
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
])

# (codepoint, risk-level) for invisible/zero-width chars
INVISIBLE_CHAR_CPS = [
    (0x200B, 'high'), (0x200C, 'high'), (0x200D, 'high'), (0x200E, 'high'),
    (0x200F, 'high'), (0xFEFF, 'high'), (0x00AD, 'med'),  (0x2060, 'high'),
    (0x2061, 'med'),  (0x2062, 'med'),  (0x2063, 'med'),  (0x2064, 'med'),
    (0x034F, 'high'), (0x115F, 'high'), (0x1160, 'high'), (0x180E, 'high'),
    (0x3164, 'high'), (0xFFA0, 'high'), (0x2028, 'med'),  (0x2029, 'med'),
    (0x00A0, 'low'),  (0x202F, 'med'),  (0xFFFD, 'med'),
]

# (foreign_char, latin_lookalike) Cyrillic + Greek homoglyphs
HOMOGLYPH_PAIRS = [
    ('а','a'),('е','e'),('о','o'),('р','p'),('с','c'),('х','x'),('у','y'),
    ('В','B'),('Е','E'),('З','3'),('К','K'),('М','M'),('Н','H'),('О','O'),
    ('Р','P'),('С','C'),('Т','T'),('Х','X'),
    ('ο','o'),('ν','v'),('Α','A'),('Β','B'),('Ε','E'),('Ζ','Z'),('Η','H'),
    ('Ι','I'),('Κ','K'),('Μ','M'),('Ν','N'),('Ο','O'),('Ρ','P'),('Τ','T'),
    ('Υ','Y'),('Χ','X'),
]

WEIGHTS = [
    0.01,  # 0  perplexity
    0.09,  # 1  burstiness
    0.08,  # 2  lexical diversity
    0.07,  # 3  AI phrases
    0.01,  # 4  hedging
    0.01,  # 5  passive voice
    0.06,  # 6  transitions
    0.02,  # 7  clause depth
    0.07,  # 8  punctuation variance
    0.10,  # 9  paragraph uniformity
    0.02,  # 10 rare words
    0.10,  # 11 register stability / formality shift
    0.04,  # 12 n-gram repetition
    0.05,  # 13 sentence opener diversity
    0.07,  # 14 punctuation fingerprint
    0.05,  # 15 vocab clustering
    0.07,  # 16 density melody
    0.08,  # 17 monte carlo
]

SIGNAL_NAMES = [
    'Perplexity', 'Burstiness', 'Lexical Div', 'AI Phrases', 'Hedging',
    'Passive', 'Transitions', 'Clause Depth', 'Punct Var', 'Para Uniform',
    'Rare Words', 'Formality', 'N-gram Rep', 'Opener Div', 'Punct Print',
    'Vocab Cluster', 'Density Melody', 'Monte Carlo',
]

COMMON_WORDS = set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with',
    'by','from','up','about','into','through','during','is','are','was','were',
    'be','been','have','has','had','do','does','did','will','would','could',
    'should','may','might','can','this','that','these','those','it','its','i',
    'you','he','she','we','they','my','your','his','her','our','their','what',
    'which','who','when','where','how','if','then','than','so','as','not','no',
    'there','here','more','also','just','only','very','much','many','most',
    'some','all','any','each','other','over',
])

CONTRACTION_RE = re.compile(
    r"\b(i'm|it's|don't|can't|won't|isn't|aren't|wasn't|weren't|you're|we're|"
    r"they're|i've|you've|we've|they've|i'd|you'd|he'd|she'd|we'd|they'd|"
    r"i'll|you'll|he'll|she'll|we'll|they'll|hasn't|haven't|hadn't|doesn't|"
    r"didn't|wouldn't|couldn't|shouldn't|that's|who's|what's|there's|here's)\b",
    re.IGNORECASE,
)

INFORMAL_RE = re.compile(
    r"\b(i'm|it's|don't|can't|won't|isn't|you're|we're|they're|gonna|wanna|"
    r"kinda|sorta|yeah|ok|okay|alright|stuff|things|get|got|pretty|really|"
    r"actually|basically|literally|honestly|like)\b",
    re.IGNORECASE,
)

PASSIVE_RE = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', re.IGNORECASE
)

SUBORD_RE = re.compile(
    r'\b(which|that|although|because|since|while|when|where|if|unless|whereas|whether)\b',
    re.IGNORECASE,
)

FINAL_PREP_RE = re.compile(
    r'\b(in|on|at|to|for|of|with|by|from|up|about|into|out|off|over|under|after|before)'
    r'\s*[.!?]\s*$',
    re.IGNORECASE,
)

# ── HELPERS ───────────────────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r"\b[a-z']+\b", text.lower()) or []

def get_sentences(text):
    return re.findall(r'[^.!?]+[.!?]+', text) or [text]

def get_paragraphs(text):
    natural = [p for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 20]
    if len(natural) >= 3:
        return natural
    sentences = [s for s in get_sentences(text) if len(s.strip()) > 10]
    if len(sentences) < 6:
        return natural if len(natural) >= 2 else [text]
    third = len(sentences) // 3
    groups = [
        ' '.join(sentences[:third]),
        ' '.join(sentences[third:third * 2]),
        ' '.join(sentences[third * 2:]),
    ]
    return [g for g in groups if len(g.strip()) > 20]

def get_natural_paragraphs(text):
    return [p for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 20]

def _cv(values):
    """Coefficient of variation for a list of numbers. Returns 0 if avg==0."""
    if not values:
        return 0
    avg = sum(values) / len(values)
    if avg == 0:
        return 0
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(var) / avg

# ── PERPLEXITY ENSEMBLE ────────────────────────────────────────────────────────

def _syllables(word):
    w = word.lower()
    w = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', w)
    w = re.sub(r'^y', '', w)
    m = re.findall(r'[aeiouy]{1,2}', w)
    return len(m) if m else 1

def calc_perplexity_proxy1(text):
    words = tokenize(text)
    if not words:
        return 0
    lengths = [len(w) for w in words]
    avg = sum(lengths) / len(lengths)
    var = sum((l - avg) ** 2 for l in lengths) / len(lengths)
    return max(0, min(100, 100 - var * 12))

def calc_perplexity_proxy2(text):
    sentences = get_sentences(text)
    if len(sentences) < 3:
        return {'score': 50, 'dataPoints': len(sentences), 'confidence': 0}
    syl_counts = []
    for s in sentences:
        ws = re.findall(r"\b[a-z']+\b", s.lower())
        syl_counts.append(sum(_syllables(w) for w in ws))
    avg = sum(syl_counts) / len(syl_counts)
    if avg == 0:
        return {'score': 50, 'dataPoints': len(sentences), 'confidence': 0}
    var = sum((s - avg) ** 2 for s in syl_counts) / len(syl_counts)
    cv = math.sqrt(var) / avg
    score = max(0, min(100, round(100 - cv * 85)))
    return {'score': score, 'dataPoints': len(sentences), 'confidence': min(len(sentences) / 10, 1.0)}

def calc_perplexity_proxy3(text):
    sentences = get_sentences(text)
    if len(sentences) < 5:
        return {'score': 50, 'confidence': 0, 'dataPoints': len(sentences)}
    lengths = [len(re.findall(r'\b\w+\b', s.strip())) for s in sentences]
    bins = [0, 0, 0, 0, 0]
    for l in lengths:
        if   l <= 7:  bins[0] += 1
        elif l <= 12: bins[1] += 1
        elif l <= 18: bins[2] += 1
        elif l <= 25: bins[3] += 1
        else:         bins[4] += 1
    total = len(lengths)
    entropy = 0.0
    for b in bins:
        if b > 0:
            p = b / total
            entropy -= p * math.log2(p)
    score = max(0, min(100, round(100 - (entropy / math.log2(5)) * 90)))
    return {'score': score, 'dataPoints': total, 'confidence': min(total / 10, 1.0)}

def calc_perplexity(text):
    p1_raw = calc_perplexity_proxy1(text)
    p1 = {'score': p1_raw, 'confidence': 0, 'dataPoints': len(tokenize(text))}  # disabled
    p2 = calc_perplexity_proxy2(text)
    p3 = calc_perplexity_proxy3(text)
    valid = [p for p in [p1, p2, p3] if p['confidence'] >= 0.2 and p['dataPoints'] >= 5]
    if not valid:
        return {'score': 50}
    total_conf = sum(p['confidence'] for p in valid)
    ws = sum(p['score'] * p['confidence'] for p in valid) / total_conf
    return {'score': round(ws)}

# ── LAYER 1 SIGNALS ───────────────────────────────────────────────────────────

def calc_burstiness(text):
    sentences = get_sentences(text)
    if len(sentences) < 3:
        return 50
    lengths = [len(s.strip().split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    if avg == 0:
        return 50
    cv = _cv(lengths)
    return max(0, min(100, 100 - cv * 90))

def calc_lexical_diversity(text):
    words = tokenize(text)
    if len(words) < 10:
        return 50
    ttr = len(set(words)) / len(words)
    return max(0, min(100, 100 - ttr * 130))

def calc_ai_phrases(text):
    lower = text.lower()
    t1 = sum(1 for p in AI_PHRASES_T1 if p in lower)
    t2 = sum(1 for p in AI_PHRASES_T2 if p in lower)
    t1s = 0 if t1 == 0 else min(95, 58 + (t1 - 1) * 12)
    t2s = 0 if t2 == 0 else (15 if t2 == 1 else min(55, 27 + (t2 - 2) * 8))
    combo = 12 if (t1 >= 1 and t2 >= 2) else (5 if (t1 >= 1 and t2 >= 1) else 0)
    return {'score': min(100, t1s + t2s + combo)}

def calc_hedging(text):
    lower = text.lower()
    words = tokenize(text)
    if not words:
        return 0
    count = sum(
        len(re.findall(r'\b' + h.replace(' ', r'\s+') + r'\b', lower))
        for h in HEDGE_WORDS
    )
    rate = count / (len(words) / 100)
    return min(100, rate * 18)

def calc_passive_voice(text):
    sentences = get_sentences(text)
    passive_count = sum(1 for s in sentences if PASSIVE_RE.search(s))
    return min(100, (passive_count / len(sentences)) * 160)

def calc_transitions(text):
    lower = text.lower()
    words = tokenize(text)
    if not words:
        return 0
    count = sum(
        len(re.findall(r'\b' + re.escape(t) + r'\b', lower))
        for t in TRANSITION_WORDS
    )
    return min(100, (count / (len(words) / 100)) * 22)

def calc_clause_depth(text):
    sentences = get_sentences(text)
    total = sum(len(SUBORD_RE.findall(s)) for s in sentences)
    avg = total / len(sentences)
    if avg > 0.8 and avg < 2.5:
        return max(0, min(100, (2.5 - abs(avg - 1.6)) * 50))
    return 30

def calc_punctuation_variance(text):
    sentences = [s for s in get_sentences(text) if len(s.strip()) > 10]
    if len(sentences) < 4:
        return 50
    counts = [len(re.findall(r'[,;:—\-()]', s)) for s in sentences]
    avg = sum(counts) / len(counts)
    if avg < 0.3:
        return 50
    cv = _cv(counts)
    return max(0, min(100, round(90 - cv * 55)))

def calc_paragraph_uniformity(text):
    paras = get_natural_paragraphs(text)
    if len(paras) < 2:
        return 50
    lengths = [len(p.strip().split()) for p in paras]
    cv = _cv(lengths)
    return max(0, min(100, 90 - cv * 80))

def calc_rare_words(text):
    words = tokenize(text)
    content = [w for w in words if w not in COMMON_WORDS and len(w) > 3]
    if not content:
        return 50
    rare_ttr = len(set(content)) / len(content)
    return max(0, min(100, 85 - rare_ttr * 90))

def calc_formality_shift(text):
    sentences = get_sentences(text)
    if len(sentences) < 4:
        return 50
    inf_scores = [len(INFORMAL_RE.findall(s)) for s in sentences]
    total_informal = sum(inf_scores)
    avg = sum(inf_scores) / len(inf_scores)
    var = sum((s - avg) ** 2 for s in inf_scores) / len(inf_scores)
    score = max(0, min(100, 75 - math.sqrt(var) * 30))
    if total_informal == 0 and len(sentences) > 5:
        score = min(score + 20, 100)
    return score

def calc_ngram_repetition(text):
    words = tokenize(text)
    if len(words) < 30:
        return 50
    trigrams = {}
    for i in range(len(words) - 2):
        k = f"{words[i]} {words[i+1]} {words[i+2]}"
        trigrams[k] = trigrams.get(k, 0) + 1
    total = len(trigrams)
    if total == 0:
        return 50
    repeated = sum(1 for v in trigrams.values() if v > 1)
    return min(100, (repeated / total) * 300)

def calc_sentence_opener_diversity(text):
    sentences = get_sentences(text)
    if len(sentences) < 5:
        return 50
    openers = []
    for s in sentences:
        ws = re.findall(r"\b[a-z']+\b", s.strip().lower())
        if ws:
            openers.append(' '.join(ws[:2]))
    if not openers:
        return 50
    diversity = len(set(openers)) / len(openers)
    return max(0, min(100, 90 - diversity * 95))

def calc_punctuation_fingerprint(text):
    sentences = get_sentences(text)
    total = len(sentences)
    if total < 4:
        return 50
    em_dash_per_sent = len(re.findall(r'—', text)) / total
    paren_rate = sum(1 for s in sentences if re.search(r'\(.*?\)', s)) / total
    ai_dash_rate = len(re.findall(r'\w\s*—\s*\w', text)) / total
    score = 0
    if   em_dash_per_sent > 0.4: score += 35
    elif em_dash_per_sent > 0.2: score += 15
    if ai_dash_rate > 0.3:       score += 25
    if 0.25 < paren_rate < 0.5:  score += 20
    if len(re.findall(r'!', text)) == 0 and total > 8:
        score += 10
    return min(100, score)

def calc_vocab_clustering(text):
    paragraphs = get_paragraphs(text)
    if len(paragraphs) < 2:
        return 50
    all_words = tokenize(text)
    freq = {}
    for w in all_words:
        if len(w) > 4:
            freq[w] = freq.get(w, 0) + 1
    key_terms = {w for w, c in freq.items() if c >= 2}
    if not key_terms:
        return 30
    densities = []
    for p in paragraphs:
        pw = tokenize(p)
        if not pw:
            densities.append(0.0)
            continue
        densities.append(sum(1 for w in pw if w in key_terms) / len(pw))
    avg = sum(densities) / len(densities)
    if avg == 0:
        return 50
    cv = _cv(densities)
    return max(0, min(100, 85 - cv * 75))

# ── DENSITY MELODY ENSEMBLE ───────────────────────────────────────────────────

def _density_word_weight(word):
    w = word.lower()
    if w in DENSITY_STOPWORDS:
        return 0.05
    l = len(w)
    if l <= 3: return 0.15
    if l <= 6: return 0.40
    if l <= 9: return 0.72
    return 1.0

def calc_density_melody(text, window_size=5, threshold=0.25, granularity='sentence'):
    curve = []

    if granularity == 'sentence':
        sents = [s for s in get_sentences(text) if len(s.strip()) > 10]
        if len(sents) < window_size + 1:
            return {'score': 50, 'dataPoints': 0, 'confidence': 0}
        sent_dens = []
        for s in sents:
            sw = re.findall(r'\b[a-z]+\b', s.lower())
            sent_dens.append(sum(_density_word_weight(w) for w in sw) / len(sw) if sw else 0)
        for i in range(len(sent_dens) - window_size + 1):
            curve.append(sum(sent_dens[i:i + window_size]) / window_size)

    else:  # chunk
        all_words = re.findall(r'\b[a-z]+\b', text.lower())
        CHUNK = 20
        wts = [_density_word_weight(w) for w in all_words]
        chunks = [
            sum(wts[i:i + CHUNK]) / CHUNK
            for i in range(0, len(wts) - CHUNK + 1, CHUNK)
        ]
        if len(chunks) < window_size + 1:
            return {'score': 50, 'dataPoints': 0, 'confidence': 0}
        for i in range(len(chunks) - window_size + 1):
            curve.append(sum(chunks[i:i + window_size]) / window_size)

    dp = len(curve)
    if dp < window_size * 2:
        return {'score': 50, 'dataPoints': 0, 'confidence': 0}
    n = dp
    mean = sum(curve) / n
    if mean < threshold:
        return {'score': 50, 'dataPoints': 0, 'confidence': 0}

    var = sum((v - mean) ** 2 for v in curve) / n
    cv = math.sqrt(var) / mean if mean > 0.01 else 0

    # Gap CV on above-mean positions
    high_pos = [i for i, v in enumerate(curve) if v > mean]
    gap_cv = 0
    if len(high_pos) >= 5:
        gaps = [high_pos[i] - high_pos[i - 1] for i in range(1, len(high_pos))]
        gm = sum(gaps) / len(gaps)
        gv = sum((g - gm) ** 2 for g in gaps) / len(gaps)
        gap_cv = math.sqrt(gv) / gm if gm > 0 else 0

    # High-frequency energy
    smooth = []
    for i in range(n):
        lo, hi = max(0, i - 2), min(n - 1, i + 2)
        vals = curve[lo:hi + 1]
        smooth.append(sum(vals) / len(vals))
    hf_energy = sum(abs(curve[i] - smooth[i]) for i in range(n)) / n

    cv_score    = max(0, min(100, round(100 - (cv - 0.10) / 0.24 * 100)))
    gap_cv_score = max(0, min(100, round(100 - (gap_cv - 0.40) / 1.0 * 100))) if gap_cv > 0 else 50
    hf_score    = max(0, min(100, round(100 - (hf_energy - 0.010) / 0.040 * 100)))

    score = round(cv_score * 0.45 + gap_cv_score * 0.30 + hf_score * 0.25)
    return {'score': score, 'dataPoints': dp, 'confidence': min(1.0, dp / 15)}

def calc_density_melody_ensemble(text):
    param_sets = [
        dict(window_size=3, threshold=0.20, granularity='sentence'),
        dict(window_size=5, threshold=0.25, granularity='sentence'),
        dict(window_size=7, threshold=0.30, granularity='sentence'),
        dict(window_size=3, threshold=0.15, granularity='chunk'),
        dict(window_size=5, threshold=0.20, granularity='chunk'),
    ]
    runs = [calc_density_melody(text, **p) for p in param_sets]
    valid = [r for r in runs if r['dataPoints'] >= 3 and r['confidence'] > 0]
    if not valid:
        return 50
    total_conf = sum(r['confidence'] for r in valid)
    return round(sum(r['score'] * r['confidence'] for r in valid) / total_conf) if total_conf else 50

# ── MONTE CARLO ───────────────────────────────────────────────────────────────

def run_monte_carlo(text):
    words = text.split()
    if len(words) < 150:
        return {'score': 50}
    window_words = min(200, max(100, len(words) // 4))
    target = min(20, max(8, len(words) // 150))
    results = []
    for _ in range(target):
        if len(words) <= window_words:
            subset = text
        else:
            start = random.randint(0, len(words) - window_words)
            subset = ' '.join(words[start:start + window_words])
        b  = calc_burstiness(subset)
        a  = calc_ai_phrases(subset)['score']
        f  = calc_formality_shift(subset)
        tr = calc_transitions(subset)
        vc = calc_vocab_clustering(subset)
        results.append(round((b + a + f + tr + vc) / 5))
    if not results:
        return {'score': 50}
    return {'score': round(sum(results) / len(results))}

# ── LAYER 2: FORENSIC CHARACTER SCAN ─────────────────────────────────────────

def run_forensic_analysis(text):
    score = 0
    # Invisible / zero-width characters
    invis_score = 0
    nbsp_count = text.count(' ')
    for cp, risk in INVISIBLE_CHAR_CPS:
        if cp == 0x00A0:
            continue  # handled separately
        if chr(cp) in text:
            if risk == 'high':
                invis_score += 15
    invis_score = min(invis_score, 60)
    score += invis_score
    if nbsp_count > 3:
        score += 15

    # Homoglyphs
    hg = min(sum(8 for f, _ in HOMOGLYPH_PAIRS if f in text), 40)
    score += hg

    # Mixed quote styles
    if ("'" in text) and ('‘' in text or '’' in text):
        score += 12
    if ('"' in text) and ('“' in text or '”' in text):
        score += 12

    return min(100, score)

# ── LAYER 5: AUTHORIAL CONSISTENCY ───────────────────────────────────────────

def calc_contraction_consistency(text):
    paras = get_paragraphs(text)
    if len(paras) < 3:
        return 50
    rates = []
    for p in paras:
        wc = len(tokenize(p)) or 1
        rates.append(len(CONTRACTION_RE.findall(p)) / wc)
    avg = sum(rates) / len(rates)
    if avg < 0.005:
        return 30
    cv = _cv(rates)
    return max(0, min(100, 88 - cv * 75))

def calc_oxford_comma_consistency(text):
    with_ox = len(re.findall(r'\b\w+\b,\s+\b\w+\b,\s+and\s+\b\w+\b', text, re.I))
    without = len(re.findall(r'\b\w+\b,\s+\b\w+\b\s+and\s+\b\w+\b', text, re.I))
    true_without = max(0, without - with_ox)
    total = with_ox + true_without
    if total < 2:
        return 50
    inconsistency = min(with_ox, true_without) / total
    return max(0, min(100, 90 - inconsistency * 180))

def calc_number_formatting_consistency(text):
    wn = len(re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b', text, re.I))
    dn = len(re.findall(r'\b([1-9]|1[0-2])\b', text))
    total = wn + dn
    if total < 3:
        return 50
    inconsistency = min(wn, dn) / total
    return max(0, min(100, 85 - inconsistency * 150))

def calc_prep_ending_consistency(text):
    paras = get_paragraphs(text)
    if len(paras) < 3:
        return 50
    rates = []
    for p in paras:
        sents = get_sentences(p)
        if not sents:
            rates.append(0)
            continue
        rates.append(sum(1 for s in sents if FINAL_PREP_RE.search(s.strip())) / len(sents))
    avg = sum(rates) / len(rates)
    var = sum((r - avg) ** 2 for r in rates) / len(rates)
    return max(0, min(100, 85 - math.sqrt(var) * 400))

def calc_paragraph_opener_consistency(text):
    paras = get_paragraphs(text)
    if len(paras) < 3:
        return 50
    articles    = {'the', 'a', 'an'}
    pronouns    = {'i','you','he','she','it','we','they','this','these','those',
                   'my','your','his','her','its','our','their'}
    transitions = {'however','therefore','thus','moreover','furthermore','additionally',
                   'nevertheless','consequently','first','second','third','finally',
                   'lastly','overall','ultimately','in','for','but','yet','while',
                   'although','since','because','despite'}

    def classify(w):
        if w in articles: return 'article'
        if w in pronouns: return 'pronoun'
        if w in transitions: return 'transition'
        return 'other'

    classes = []
    for p in paras:
        ws = re.findall(r'\b[a-z]+\b', p.strip().lower())
        classes.append(classify(ws[0]) if ws else 'other')

    freq = {}
    for c in classes:
        freq[c] = freq.get(c, 0) + 1
    dominant = max(freq.values())
    consistency = dominant / len(classes)
    return max(0, min(100, (consistency - 0.45) / 0.55 * 100))

def run_authoral_consistency(text):
    c = calc_contraction_consistency(text)
    o = calc_oxford_comma_consistency(text)
    n = calc_number_formatting_consistency(text)
    p = calc_prep_ending_consistency(text)
    op = calc_paragraph_opener_consistency(text)
    return round(c * 0.25 + o * 0.20 + n * 0.20 + p * 0.15 + op * 0.20)

# ── FULL SCORING PIPELINE ─────────────────────────────────────────────────────

def analyze_text(text, use_monte_carlo=True, use_density=True):
    wc = len(tokenize(text))

    # Text reliability factor (mirrors index.html confidence penalty)
    if   wc < 100: reliability = 0.60
    elif wc < 300: reliability = 0.82
    elif wc < 600: reliability = 0.92
    else:          reliability = 1.00

    perplexity    = calc_perplexity(text)
    burstiness    = calc_burstiness(text)
    lexical       = calc_lexical_diversity(text)
    ai_phrases    = calc_ai_phrases(text)
    hedging       = calc_hedging(text)
    passive       = calc_passive_voice(text)
    transitions   = calc_transitions(text)
    clause_depth  = calc_clause_depth(text)
    punct_var     = calc_punctuation_variance(text)
    para_uniform  = calc_paragraph_uniformity(text)
    rare_words    = calc_rare_words(text)
    formality     = calc_formality_shift(text)
    ngram_rep     = calc_ngram_repetition(text)
    opener_div    = calc_sentence_opener_diversity(text)
    punct_finger  = calc_punctuation_fingerprint(text)
    vocab_cluster = calc_vocab_clustering(text)
    density       = calc_density_melody_ensemble(text) if use_density else 50
    mc            = run_monte_carlo(text) if use_monte_carlo else {'score': 50}

    scores = [
        perplexity['score'],   # 0
        burstiness,            # 1
        lexical,               # 2
        ai_phrases['score'],   # 3
        hedging,               # 4
        passive,               # 5
        transitions,           # 6
        clause_depth,          # 7
        punct_var,             # 8
        para_uniform,          # 9
        rare_words,            # 10
        formality,             # 11
        ngram_rep,             # 12
        opener_div,            # 13
        punct_finger,          # 14
        vocab_cluster,         # 15
        density,               # 16
        mc['score'],           # 17
    ]

    # Evidence accumulation (identical to index.html)
    evidence = sum(max(0, s - 50) * WEIGHTS[i] * reliability for i, s in enumerate(scores))
    base_composite = min(100, round(evidence * 5))

    # Convergence bonus: anchor signals at idx 1,3,11,15
    anchors_hot = sum(1 for i in [1, 3, 11, 15] if scores[i] > 70)
    conv_bonus = [0, 0, 5, 12, 18][min(anchors_hot, 4)]

    # Smoking gun bonus on AI phrases
    smoking_gun = 30 if scores[3] >= 100 else (15 if scores[3] > 85 else 0)

    composite = min(100, base_composite + conv_bonus + smoking_gun)

    forensic = run_forensic_analysis(text)
    layer5   = run_authoral_consistency(text)

    combined = round(min(100, composite * 0.75 + forensic * 0.15 + layer5 * 0.10))

    if   combined >= 50: verdict = 'ai'
    elif combined >= 20: verdict = 'ambiguous'
    else:                verdict = 'human'

    return {
        'score':     combined,
        'composite': composite,
        'forensic':  forensic,
        'layer5':    layer5,
        'verdict':   verdict,
        'wc':        wc,
        'scores':    scores,
    }

# ── METRICS & REPORTING ───────────────────────────────────────────────────────

def compute_metrics(rows):
    """
    Binary classification: AI vs human.
    "ambiguous" counts as a positive (AI) prediction — conservative.
    strict_accuracy treats ambiguous as wrong for both labels.
    """
    if not rows:
        return None
    tp = sum(1 for r in rows if r['label'] == 'ai'    and r['verdict'] in ('ai', 'ambiguous'))
    fp = sum(1 for r in rows if r['label'] == 'human' and r['verdict'] in ('ai', 'ambiguous'))
    tn = sum(1 for r in rows if r['label'] == 'human' and r['verdict'] == 'human')
    fn = sum(1 for r in rows if r['label'] == 'ai'    and r['verdict'] == 'human')

    acc  = (tp + tn) / len(rows)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    strict_tp = sum(1 for r in rows if r['label'] == 'ai'    and r['verdict'] == 'ai')
    strict_tn = sum(1 for r in rows if r['label'] == 'human' and r['verdict'] == 'human')
    strict_acc = (strict_tp + strict_tn) / len(rows)

    return {
        'n': len(rows), 'accuracy': acc, 'strict_accuracy': strict_acc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'ambiguous': sum(1 for r in rows if r['verdict'] == 'ambiguous'),
        'avg_score': sum(r['score'] for r in rows) / len(rows),
    }

def fmt_row(label, m, indent=2):
    pad = ' ' * indent
    return (f"{pad}{label:<28} {m['n']:>5}  {m['accuracy']:>6.1%}  {m['strict_accuracy']:>7.1%}  "
            f"{m['precision']:>6.1%}  {m['recall']:>6.1%}  {m['f1']:>6.1%}  {m['avg_score']:>7.1f}")

def print_breakdown(title, rows, key_fn, min_n=5):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    W = 80
    print(f"\n{'─'*W}")
    print(f"  {title}")
    print(f"{'─'*W}")
    print(f"  {'Category':<28} {'n':>5}  {'Acc':>6}  {'Strict':>7}  "
          f"{'P':>6}  {'R':>6}  {'F1':>6}  {'AvgScr':>7}")
    print(f"  {'─'*74}")
    for k, group in sorted(groups.items(), key=lambda x: -compute_metrics(x[1])['f1']):
        m = compute_metrics(group)
        if m and m['n'] >= min_n:
            print(fmt_row(str(k)[:27], m))

def print_score_distribution(rows):
    W = 80
    print(f"\n{'─'*W}")
    print(f"  Score Distribution (buckets of 10)")
    print(f"{'─'*W}")
    buckets = defaultdict(lambda: {'ai': 0, 'human': 0})
    for r in rows:
        buckets[(r['score'] // 10) * 10][r['label']] += 1
    print(f"  {'Range':<10}  {'AI texts':>10}  {'Human texts':>12}  {'Total':>7}")
    print(f"  {'─'*44}")
    for b in range(0, 110, 10):
        row = buckets[b]
        tot = row['ai'] + row['human']
        if tot > 0:
            print(f"  {b:2d}–{min(b+9,100):2d}%      {row['ai']:>10}  {row['human']:>12}  {tot:>7}")

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Evaluate AI Origin Detector against the RAID benchmark'
    )
    ap.add_argument('--sample',         type=int,   default=2000)
    ap.add_argument('--split',          default='train', choices=['train', 'test'])
    ap.add_argument('--no-monte-carlo', action='store_true')
    ap.add_argument('--no-density',     action='store_true')
    ap.add_argument('--domain',         default=None)
    ap.add_argument('--attack',         default=None)
    ap.add_argument('--min-words',      type=int,   default=50)
    ap.add_argument('--seed',           type=int,   default=42)
    ap.add_argument('--dataset',        default='liamdugan/raid',
                    help='HuggingFace dataset name (default: liamdugan/raid)')
    args = ap.parse_args()

    random.seed(args.seed)
    use_mc = not args.no_monte_carlo
    use_dm = not args.no_density

    W = 80
    print(f"\n{'═'*W}")
    print(f"  AI ORIGIN DETECTOR — RAID BENCHMARK EVALUATION")
    print(f"{'═'*W}")
    print(f"  Dataset: {args.dataset}  |  Split: {args.split}  |  "
          f"Sample: {args.sample or 'all'}  |  seed={args.seed}")
    print(f"  Monte Carlo: {'ON' if use_mc else 'OFF'}  |  "
          f"Density Melody: {'ON' if use_dm else 'OFF'}  |  "
          f"min-words: {args.min_words}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\n  Loading dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  ERROR: install with:  pip install datasets huggingface_hub")
        sys.exit(1)

    try:
        ds = load_dataset(args.dataset, split=args.split, trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        print(f"\n  Common fixes:")
        print(f"    pip install datasets huggingface_hub")
        print(f"    huggingface-cli login   (if dataset requires auth)")
        print(f"    --dataset liamdugan/raid   or   --dataset raid-bench/raid")
        sys.exit(1)

    print(f"  Loaded {len(ds)} rows from '{args.dataset}' ({args.split})")

    # ── Filter ────────────────────────────────────────────────────────────────
    rows = list(ds)

    # Map any boolean-ish repetition_penalty to string
    for r in rows:
        rp = r.get('repetition_penalty')
        if isinstance(rp, bool):
            r['repetition_penalty'] = 'yes' if rp else 'no'

    if args.domain:
        rows = [r for r in rows if r.get('domain') == args.domain]
        print(f"  domain='{args.domain}' → {len(rows)} rows")
    if args.attack:
        rows = [r for r in rows if r.get('attack') == args.attack]
        print(f"  attack='{args.attack}' → {len(rows)} rows")

    rows = [r for r in rows
            if len(tokenize(r.get('generation', ''))) >= args.min_words]
    print(f"  After min-words ({args.min_words}+): {len(rows)} rows")

    if not rows:
        print("  No rows remaining — check your filters.")
        sys.exit(1)

    # ── Stratified sample ─────────────────────────────────────────────────────
    if args.sample and args.sample < len(rows):
        ai_pool    = [r for r in rows if r.get('label') == 'ai']
        human_pool = [r for r in rows if r.get('label') == 'human']
        n_each = min(args.sample // 2, len(ai_pool), len(human_pool))
        rows = random.sample(ai_pool, n_each) + random.sample(human_pool, n_each)
        random.shuffle(rows)
        print(f"  Stratified sample: {len(rows)} ({n_each} AI + {n_each} human)")
    else:
        print(f"  Using full filtered set: {len(rows)} rows")

    # ── Analyse ───────────────────────────────────────────────────────────────
    print(f"\n  Running detection on {len(rows)} texts...")
    results = []
    for i, row in enumerate(rows):
        if i > 0 and i % 200 == 0:
            pct = i / len(rows) * 100
            print(f"    {i}/{len(rows)} ({pct:.0f}%)  ", end='', flush=True)
            done = results[:i]
            m = compute_metrics(done)
            print(f"running acc={m['accuracy']:.1%}  F1={m['f1']:.1%}")

        text   = row.get('generation', '')
        label  = row.get('label', 'unknown')
        result = analyze_text(text, use_monte_carlo=use_mc, use_density=use_dm)
        result.update({
            'label':             label,
            'model':             row.get('model', 'unknown'),
            'domain':            row.get('domain', 'unknown'),
            'attack':            row.get('attack', 'none'),
            'decoding':          row.get('decoding', 'unknown'),
            'repetition_penalty': str(row.get('repetition_penalty', 'unknown')),
        })
        results.append(result)

    # ── Overall ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  OVERALL  (n={len(results)})")
    print(f"{'═'*W}")
    overall = compute_metrics(results)
    print(f"  Accuracy      : {overall['accuracy']:.1%}  "
          f"(strict: {overall['strict_accuracy']:.1%})")
    print(f"  Precision     : {overall['precision']:.1%}")
    print(f"  Recall        : {overall['recall']:.1%}")
    print(f"  F1            : {overall['f1']:.1%}")
    print(f"  Ambiguous     : {overall['ambiguous']} "
          f"({overall['ambiguous']/len(results)*100:.1f}%)")

    ai_rows    = [r for r in results if r['label'] == 'ai']
    human_rows = [r for r in results if r['label'] == 'human']

    if ai_rows:
        detected  = sum(1 for r in ai_rows if r['verdict'] == 'ai')
        ambig_ai  = sum(1 for r in ai_rows if r['verdict'] == 'ambiguous')
        missed    = sum(1 for r in ai_rows if r['verdict'] == 'human')
        avg_ai    = sum(r['score'] for r in ai_rows) / len(ai_rows)
        print(f"\n  AI texts   n={len(ai_rows):4d}  avg_score={avg_ai:.1f}"
              f"  detected={detected}  ambiguous={ambig_ai}  missed={missed}")

    if human_rows:
        fp        = sum(1 for r in human_rows if r['verdict'] == 'ai')
        ambig_hu  = sum(1 for r in human_rows if r['verdict'] == 'ambiguous')
        correct   = sum(1 for r in human_rows if r['verdict'] == 'human')
        avg_hu    = sum(r['score'] for r in human_rows) / len(human_rows)
        print(f"  Human texts n={len(human_rows):4d}  avg_score={avg_hu:.1f}"
              f"  FP={fp}  ambiguous={ambig_hu}  correct={correct}")

    # ── Breakdowns ────────────────────────────────────────────────────────────
    print_breakdown("BY AI MODEL",           results, lambda r: r['model'])
    print_breakdown("BY DOMAIN",             results, lambda r: r['domain'])
    print_breakdown("BY ATTACK TYPE",        results, lambda r: r['attack'])
    print_breakdown("BY DECODING STRATEGY",  results, lambda r: r['decoding'], min_n=3)
    print_breakdown("BY REPETITION PENALTY", results, lambda r: r['repetition_penalty'], min_n=3)

    # ── Score distribution ────────────────────────────────────────────────────
    print_score_distribution(results)

    # ── Per-signal analysis ───────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  SIGNAL ANALYSIS — mean score by label  (higher = more AI-like)")
    print(f"{'─'*W}")
    print(f"  {'Signal':<20}  {'Wt':>5}  {'AI avg':>8}  {'Hu avg':>8}  {'Gap':>7}  {'Useful?':>8}")
    print(f"  {'─'*60}")
    for i, name in enumerate(SIGNAL_NAMES):
        ai_avg = sum(r['scores'][i] for r in ai_rows)  / len(ai_rows)  if ai_rows  else 0
        hu_avg = sum(r['scores'][i] for r in human_rows)/ len(human_rows) if human_rows else 0
        gap    = ai_avg - hu_avg
        useful = 'YES' if abs(gap) >= 5 else 'low'
        print(f"  {name:<20}  {WEIGHTS[i]:>5.2f}  {ai_avg:>8.1f}  {hu_avg:>8.1f}  "
              f"{gap:>+7.1f}  {useful:>8}")

    # ── False positive / negative deep-dives ─────────────────────────────────
    fp_rows = [r for r in human_rows if r['verdict'] == 'ai']
    fn_rows = [r for r in ai_rows    if r['verdict'] == 'human']

    print(f"\n{'─'*W}")
    print(f"  FALSE POSITIVES — human texts flagged as AI  (n={len(fp_rows)}, "
          f"{len(fp_rows)/len(human_rows)*100:.1f}% of human)")
    print(f"{'─'*W}")
    if fp_rows:
        by_domain = defaultdict(int)
        for r in fp_rows: by_domain[r['domain']] += 1
        for d, c in sorted(by_domain.items(), key=lambda x: -x[1]):
            print(f"  {d:<22} {c:>5}  ({c/len(fp_rows)*100:.0f}%)")
    else:
        print("  None.")

    print(f"\n{'─'*W}")
    print(f"  FALSE NEGATIVES — AI texts missed  (n={len(fn_rows)}, "
          f"{len(fn_rows)/len(ai_rows)*100:.1f}% of AI)")
    print(f"{'─'*W}")
    if fn_rows:
        by_model  = defaultdict(int)
        by_attack = defaultdict(int)
        for r in fn_rows:
            by_model[r['model']]   += 1
            by_attack[r['attack']] += 1
        print(f"  By model:")
        for m, c in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"    {m:<22} {c:>5}")
        print(f"  By attack:")
        for a, c in sorted(by_attack.items(), key=lambda x: -x[1]):
            print(f"    {a:<22} {c:>5}")
    else:
        print("  None.")

    print(f"\n{'═'*W}\n")


if __name__ == '__main__':
    main()
