"""
Fast-DetectGPT — conditional probability curvature (Bao et al., ICLR 2024).

Zero-shot AI-text detection. Unlike a trained classifier (which we just proved
fails to generalize across datasets — LOSO AUROC 0.52), this measures an
intrinsic property of the text under a language model and has no learned
decision boundary, so it should transfer across domains/generators.

Single-model variant: the SAME model (gpt2, 124M) both scores the text and
provides the sampling distribution. For each token position i the model gives a
probability distribution p_i over the vocabulary. We compute, per position:

    lp_i      = log p_i[x_i]                      # log-prob of the ACTUAL token
    mu_i      = E_{v~p_i}[log p_i[v]]             # expected log-prob  (= -entropy)
    var_i     = Var_{v~p_i}[log p_i[v]]           # variance of log-prob

Summing over positions gives the conditional curvature ("sampling discrepancy"):

    d(x) = ( sum_i lp_i  -  sum_i mu_i ) / sqrt( sum_i var_i )

Human text sits near what the model expects (d ~ 0); machine text is unusually
high-probability under the model (d large positive). Threshold d to decide.

This module is import-safe: app.py (HF Space) and eval_fastdetect.py both call
`fast_detect_score(text)`. The model loads once on first call.
"""

import math
from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "gpt2"          # 124M — runs on the free HF Space CPU tier
MAX_TOKENS = 1024           # gpt2 context window; longer text is truncated
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load():
    """Load tokenizer + model once and cache. Returns (tokenizer, model)."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()
    return tok, model


@torch.no_grad()
def fast_detect_score(text: str):
    """
    Compute the Fast-DetectGPT conditional curvature for `text`.

    Returns a dict:
        { "discrepancy": float,   # raw d(x); higher = more machine-like
          "n_tokens": int,        # tokens actually scored
          "ok": bool }            # False if text too short to score
    """
    tok, model = _load()

    ids = tok(text, return_tensors="pt", truncation=True,
              max_length=MAX_TOKENS).input_ids.to(DEVICE)
    n = ids.shape[1]
    if n < 5:                                   # need a few tokens for a signal
        return {"discrepancy": 0.0, "n_tokens": n, "ok": False}

    logits = model(ids).logits                  # [1, seq, vocab]

    # Predict token i from positions < i: align logits[:-1] with targets[1:].
    shift_logits = logits[0, :-1, :]            # [seq-1, vocab]
    shift_targets = ids[0, 1:]                  # [seq-1]

    log_probs = torch.log_softmax(shift_logits, dim=-1)   # [seq-1, vocab]
    probs = log_probs.exp()

    # lp_i: log-prob assigned to the actual next token at each position
    lp = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)  # [seq-1]

    # mu_i = sum_v p*log p  (expected log-prob under the model's own dist)
    mu = (probs * log_probs).sum(dim=-1)                    # [seq-1]
    # E[(log p)^2] = sum_v p*(log p)^2 ; var = E[(logp)^2] - mu^2
    e_sq = (probs * log_probs.pow(2)).sum(dim=-1)           # [seq-1]
    var = (e_sq - mu.pow(2)).clamp(min=0)                   # numerical floor

    num = (lp.sum() - mu.sum())
    den = var.sum().sqrt()
    d = (num / den).item() if den.item() > 0 else 0.0

    return {"discrepancy": d, "n_tokens": int(shift_targets.shape[0]), "ok": True}


if __name__ == "__main__":
    # Quick smoke test
    samples = [
        ("human-ish", "i mean honestly the bus was late again and i just stood "
                       "there freezing, kinda regretting not driving tbh."),
        ("ai-ish", "Furthermore, it is important to note that effective time "
                    "management plays a crucial role in achieving one's goals "
                    "and maintaining a balanced lifestyle overall."),
    ]
    for name, t in samples:
        r = fast_detect_score(t)
        print(f"{name:10s}  d={r['discrepancy']:+.3f}  tokens={r['n_tokens']}")
