"""
Python port of the AI Layered Detection Tool (github.com/Matman22/Ai-Layered-Detection-Tool).
Mirrors the JS scoring logic: Layer 1 (linguistic, 75%) + Layer 2 (forensic, 15%) + Layer 5 (authorial, 10%).
Returns a score 0-100 where >= 60 = AI, 40-59 = Mixed, < 40 = Human.
"""

import re
import math
import unicodedata
from collections import Counter


# ---------------------------------------------------------------------------
# AI phrase lists (ported from JS)
# ---------------------------------------------------------------------------
AI_PHRASES_T1 = [
    "it is worth noting", "delve into", "multifaceted", "at its core",
    "in the realm of", "as we explore", "tapestry", "underscore", "pivotal",
    "embodies", "underpins", "first and foremost", "in today's world",
    "in today's fast-paced", "one can argue", "it's important to note",
    "it is important to note", "in this essay", "in summary", "in conclusion",
    "it goes without saying", "needless to say", "the fact of the matter",
]

AI_PHRASES_T2 = [
    "furthermore", "moreover", "in conclusion", "leveraging", "foster",
    "paradigm", "robust", "streamline", "synergy", "utilize", "navigating",
    "holistic", "nuanced", "when it comes to", "best practices",
    "game-changer", "cutting-edge", "proactive", "stakeholders",
    "innovative", "comprehensive", "dynamic", "actionable", "scalable",
    "impactful", "deep dive", "think outside the box", "move the needle",
    "circle back", "bandwidth", "ecosystem", "disruptive", "transformative",
]

DENSITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "it", "its",
    "that", "this", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "our", "their",
    "not", "no", "so", "if", "as", "up", "out", "about", "into", "through",
}

INFORMAL_RE = re.compile(
    r"\b(i'm|it's|don't|can't|won't|isn't|aren't|wasn't|weren't|you're"
    r"|we're|they're|gonna|wanna|kinda|yeah|ok|okay|alright|stuff|things"
    r"|pretty|really|actually|basically|literally|honestly|like)\b",
    re.IGNORECASE,
)
FORMAL_RE = re.compile(
    r"\b(therefore|subsequently|notwithstanding|whilst|regarding|pertaining"
    r"|aforementioned|henceforth|hereby|therein|pursuant|whereby)\b",
    re.IGNORECASE,
)

CONTRACTIONS_RE = re.compile(
    r"\b(i'm|it's|don't|can't|won't|isn't|aren't|wasn't|weren't|you're"
    r"|we're|they're|i've|i'd|i'll|you'd|you'll|he's|she's|that's|what's"
    r"|there's|here's|let's|who's)\b",
    re.IGNORECASE,
)

TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "additionally", "consequently", "meanwhile", "subsequently", "thus",
    "hence", "accordingly", "nonetheless", "otherwise", "conversely",
    "in contrast", "as a result", "in addition", "for example", "for instance",
    "in conclusion", "to summarize", "in summary",
}

PASSIVE_RE = re.compile(
    r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", re.IGNORECASE
)

INVISIBLE_CODEPOINTS = [
    0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD, 0x2060, 0x180E,
    0x3164, 0x115F, 0x1160,
]
HOMOGLYPHS = {
    "а": "a", "е": "e", "о": "o", "р": "p",
    "с": "c", "х": "x", "і": "i", "у": "y",
    "Α": "A", "Ε": "E", "Ι": "I", "Ο": "O",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def get_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def get_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def syllable_count(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _cv(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    if avg == 0:
        return 0.0
    var = sum((v - avg) ** 2 for v in values) / len(values)
    return math.sqrt(var) / avg


# ---------------------------------------------------------------------------
# Layer 1: Linguistic signals
# ---------------------------------------------------------------------------

def calc_burstiness(text: str) -> float:
    sentences = get_sentences(text)
    if len(sentences) < 2:
        return 50.0
    lengths = [len(s.split()) for s in sentences]
    cv = _cv(lengths)
    return max(0.0, min(100.0, 100 - cv * 90))


def calc_lexical_diversity(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 50.0
    ttr = len(set(words)) / len(words)
    return max(0.0, min(100.0, 100 - ttr * 130))


def calc_ai_phrases(text: str) -> float:
    lower = text.lower()
    t1 = sum(1 for p in AI_PHRASES_T1 if p in lower)
    t2 = sum(1 for p in AI_PHRASES_T2 if p in lower)
    t1_score = 0 if t1 == 0 else min(95, 58 + (t1 - 1) * 12)
    t2_score = 0 if t2 == 0 else (15 if t2 == 1 else min(55, 27 + (t2 - 2) * 8))
    combo = min(10, t1 + t2) if (t1 > 0 and t2 > 0) else 0
    return min(100.0, t1_score + t2_score + combo)


def calc_hedging(text: str) -> float:
    hedges = re.findall(
        r"\b(perhaps|maybe|possibly|might|could|seems|appears|arguably"
        r"|presumably|supposedly|allegedly|apparently)\b",
        text, re.IGNORECASE,
    )
    words = tokenize(text)
    if not words:
        return 0.0
    density = len(hedges) / len(words)
    return min(100.0, density * 1500)


def calc_passive_voice(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    passive = sum(1 for s in sentences if PASSIVE_RE.search(s))
    rate = passive / len(sentences)
    return min(100.0, max(0.0, (rate - 0.15) * 200))


def calc_transitions(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 50.0
    hits = [
        any(tw in s.lower() for tw in TRANSITION_WORDS)
        for s in sentences
    ]
    rate = sum(hits) / len(sentences)
    # Uniform high transition usage = AI signal
    return min(100.0, max(0.0, rate * 120 - 10))


def calc_punctuation_variance(text: str) -> float:
    sentences = get_sentences(text)
    if len(sentences) < 3:
        return 50.0
    counts = [len(re.findall(r"[,;:\-—]", s)) for s in sentences]
    cv = _cv(counts)
    # Low variance = AI (uniform punctuation density)
    return max(0.0, min(100.0, 100 - cv * 80))


def calc_paragraph_uniformity(text: str) -> float:
    paragraphs = get_paragraphs(text)
    if len(paragraphs) < 2:
        return 50.0
    lengths = [len(p.split()) for p in paragraphs]
    cv = _cv(lengths)
    return max(0.0, min(100.0, 100 - cv * 70))


def calc_rare_words(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 0.0
    long_words = [w for w in words if len(w) >= 9]
    rate = len(long_words) / len(words)
    # AI tends to use more rare/long words uniformly
    return min(100.0, rate * 250)


def calc_formality_shift(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 50.0
    inf_counts = [len(INFORMAL_RE.findall(s)) for s in sentences]
    total_informal = sum(inf_counts)
    avg = sum(inf_counts) / len(inf_counts)
    var = sum((c - avg) ** 2 for c in inf_counts) / len(inf_counts)
    score = max(0.0, min(100.0, 75 - math.sqrt(var) * 30))
    if total_informal == 0 and len(sentences) > 5:
        score = min(score + 20, 100.0)
    return score


def calc_ngram_repetition(text: str) -> float:
    words = tokenize(text)
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    counts = Counter(trigrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    rate = repeated / len(trigrams)
    return min(100.0, rate * 300)


def calc_sentence_opener_diversity(text: str) -> float:
    sentences = get_sentences(text)
    if len(sentences) < 3:
        return 50.0
    first_words = [s.split()[0].lower() for s in sentences if s.split()]
    ttr = len(set(first_words)) / len(first_words)
    # Low diversity = AI (lots of "The", "This", "In", etc.)
    return max(0.0, min(100.0, 100 - ttr * 100))


def calc_punctuation_fingerprint(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    em_dashes = len(re.findall(r"—", text))
    em_per_sent = em_dashes / len(sentences)
    paren_rate = sum(1 for s in sentences if re.search(r"\(.*?\)", s)) / len(sentences)
    ai_dash_rate = len(re.findall(r"\w\s*—\s*\w", text)) / len(sentences)
    exclamations = len(re.findall(r"!", text))

    score = 0.0
    if em_per_sent > 0.4:
        score += 35
    elif em_per_sent > 0.2:
        score += 15
    if ai_dash_rate > 0.3:
        score += 25
    if 0.25 < paren_rate < 0.5:
        score += 20
    if exclamations == 0 and len(sentences) > 8:
        score += 10
    return min(100.0, score)


def calc_vocab_clustering(text: str) -> float:
    paragraphs = get_paragraphs(text)
    if len(paragraphs) < 2:
        return 50.0
    all_words = tokenize(text)
    key_terms = {w for w, c in Counter(all_words).most_common(20) if len(w) > 5}
    if not key_terms:
        return 50.0
    rates = []
    for p in paragraphs:
        pw = tokenize(p)
        if not pw:
            continue
        rate = sum(1 for w in pw if w in key_terms) / len(pw)
        rates.append(rate)
    if len(rates) < 2:
        return 50.0
    cv = _cv(rates)
    # Uniform distribution of key terms = AI
    return max(0.0, min(100.0, 100 - cv * 80))


def calc_perplexity_proxy(text: str) -> float:
    words = tokenize(text)
    if not words:
        return 50.0
    lengths = [len(w) for w in words]
    cv1 = _cv(lengths)

    sentences = get_sentences(text)
    syllables = [sum(syllable_count(w) for w in s.split()) for s in sentences]
    cv2 = _cv(syllables)

    buckets = [0] * 5
    sent_lengths = [len(s.split()) for s in sentences]
    for sl in sent_lengths:
        idx = min(4, sl // 8)
        buckets[idx] += 1
    total = sum(buckets)
    entropy = 0.0
    if total > 0:
        for b in buckets:
            if b > 0:
                p = b / total
                entropy -= p * math.log2(p)
    # Higher entropy = more human-like variation
    entropy_score = max(0.0, min(100.0, 100 - entropy * 25))
    return (max(0.0, min(100.0, 100 - cv1 * 90)) + max(0.0, min(100.0, 100 - cv2 * 90)) + entropy_score) / 3


def calc_density_melody(text: str, window_size: int = 25) -> float:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    if len(words) < window_size * 2:
        return 50.0

    def word_weight(w: str) -> float:
        if w in DENSITY_STOPWORDS:
            return 0.05
        n = len(w)
        if n <= 3:
            return 0.15
        if n <= 6:
            return 0.40
        if n <= 9:
            return 0.72
        return 1.0

    weights = [word_weight(w) for w in words]
    curve = []
    for i in range(len(weights) - window_size):
        window = weights[i:i + window_size]
        curve.append(sum(window) / window_size)

    if len(curve) < 2:
        return 50.0
    cv = _cv(curve)
    return max(0.0, min(100.0, 100 - cv * 200))


# ---------------------------------------------------------------------------
# Layer 2: Forensic character analysis
# ---------------------------------------------------------------------------

def calc_forensic_score(text: str) -> float:
    score = 0.0
    # Invisible characters
    for cp in INVISIBLE_CODEPOINTS:
        if chr(cp) in text:
            score += 20
    # Homoglyphs
    for ch in HOMOGLYPHS:
        if ch in text:
            score += 25
    # Quote mixing
    has_smart = bool(re.search(r"[""'']", text))
    has_straight = bool(re.search(r'["\']', text))
    if has_smart and has_straight:
        score += 15
    # Non-standard spaces
    nbs = len(re.findall(r"[  　  ]", text))
    if nbs > 0:
        score += min(30, nbs * 5)
    return min(100.0, score)


# ---------------------------------------------------------------------------
# Layer 5: Authorial consistency
# ---------------------------------------------------------------------------

def calc_contraction_consistency(text: str) -> float:
    paragraphs = get_paragraphs(text)
    if len(paragraphs) < 2:
        return 50.0
    rates = []
    for p in paragraphs:
        words = tokenize(p)
        if not words:
            continue
        rate = len(CONTRACTIONS_RE.findall(p)) / len(words)
        rates.append(rate)
    if len(rates) < 2:
        return 50.0
    avg = sum(rates) / len(rates)
    if avg == 0:
        # Zero contractions throughout = AI signal
        return 80.0
    var = sum((r - avg) ** 2 for r in rates) / len(rates)
    cv = math.sqrt(var) / avg
    return max(0.0, min(100.0, 88 - cv * 75))


def calc_oxford_comma_consistency(text: str) -> float:
    with_oxford = len(re.findall(r"\b\w+\b,\s+\b\w+\b,\s+and\s+\b\w+\b", text, re.IGNORECASE))
    without_oxford = max(0, len(re.findall(r"\b\w+\b,\s+\b\w+\b\s+and\s+\b\w+\b", text, re.IGNORECASE)) - with_oxford)
    total = with_oxford + without_oxford
    if total < 2:
        return 50.0
    minority = min(with_oxford, without_oxford)
    return max(0.0, min(100.0, 90 - (minority / total * 180)))


def calc_authorial_score(text: str) -> float:
    contraction = calc_contraction_consistency(text)
    oxford = calc_oxford_comma_consistency(text)
    return (contraction * 0.6 + oxford * 0.4)


# ---------------------------------------------------------------------------
# Combined scorer
# ---------------------------------------------------------------------------

WEIGHTS = [
    0.01,  # perplexity proxy
    0.11,  # burstiness            (+0.02 — top-5 failing, reliable on hard models)
    0.06,  # lexical diversity     (-0.02 — FN scores near human baseline; signal fires but too weakly)
    0.07,  # AI phrases
    0.01,  # hedging
    0.01,  # passive voice
    0.03,  # transitions           (-0.03 — inverted on MPT/Llama-chat; actively hurts score)
    0.02,  # clause depth (omitted, using punctuation variance as proxy)
    0.07,  # punctuation variance
    0.12,  # paragraph uniformity  (+0.02 — consistently separates TP from FN)
    0.02,  # rare words
    0.12,  # formality shift       (+0.02 — top reliable signal across all model families)
    0.04,  # n-gram repetition
    0.05,  # sentence opener diversity
    0.07,  # punctuation fingerprint
    0.07,  # vocab clustering      (+0.02 — top-5 failing, reliable on hard models)
    0.07,  # density melody
    0.05,  # (reserved — placeholder equals avg of other signals)
]

assert abs(sum(WEIGHTS) - 1.0) < 1e-6, f"Weights must sum to 1, got {sum(WEIGHTS)}"


def score_text(text: str) -> dict:
    """
    Returns a dict with individual signal scores and the combined verdict.
    combined_score >= 60 → AI, 40-59 → Mixed, < 40 → Human
    prediction is 1 (AI) or 0 (Human), with Mixed counted as 1 for binary eval.
    """
    if not text or len(text.split()) < 20:
        return {"combined_score": 50.0, "prediction": 0, "label": "Insufficient text"}

    signals = [
        calc_perplexity_proxy(text),
        calc_burstiness(text),
        calc_lexical_diversity(text),
        calc_ai_phrases(text),
        calc_hedging(text),
        calc_passive_voice(text),
        calc_transitions(text),
        calc_punctuation_variance(text),   # proxy for clause depth slot
        calc_punctuation_variance(text),
        calc_paragraph_uniformity(text),
        calc_rare_words(text),
        calc_formality_shift(text),
        calc_ngram_repetition(text),
        calc_sentence_opener_diversity(text),
        calc_punctuation_fingerprint(text),
        calc_vocab_clustering(text),
        calc_density_melody(text),
        0.0,  # placeholder; replaced below
    ]
    # Use mean of other signals for the reserved slot
    signals[-1] = sum(signals[:-1]) / (len(signals) - 1)

    l1_score = sum(w * s for w, s in zip(WEIGHTS, signals))
    l2_score = calc_forensic_score(text)
    l5_score = calc_authorial_score(text)

    combined = l1_score * 0.75 + l2_score * 0.15 + l5_score * 0.10

    if combined >= 60:
        label = "AI"
        prediction = 1
    elif combined >= 40:
        label = "Mixed"
        prediction = 1  # treat Mixed as AI for binary classification
    else:
        label = "Human"
        prediction = 0

    return {
        "combined_score": round(combined, 2),
        "l1_score": round(l1_score, 2),
        "l2_score": round(l2_score, 2),
        "l5_score": round(l5_score, 2),
        "prediction": prediction,
        "label": label,
        "signals": {
            "perplexity_proxy": round(signals[0], 2),
            "burstiness": round(signals[1], 2),
            "lexical_diversity": round(signals[2], 2),
            "ai_phrases": round(signals[3], 2),
            "hedging": round(signals[4], 2),
            "passive_voice": round(signals[5], 2),
            "transitions": round(signals[6], 2),
            "punctuation_variance": round(signals[8], 2),
            "paragraph_uniformity": round(signals[9], 2),
            "rare_words": round(signals[10], 2),
            "formality_shift": round(signals[11], 2),
            "ngram_repetition": round(signals[12], 2),
            "sentence_opener_diversity": round(signals[13], 2),
            "punctuation_fingerprint": round(signals[14], 2),
            "vocab_clustering": round(signals[15], 2),
            "density_melody": round(signals[16], 2),
        },
    }
