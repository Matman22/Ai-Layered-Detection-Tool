"""
Fast-DetectGPT scoring service — FastAPI app for a free HuggingFace Space.

Exposes a clean JSON endpoint that index.html (served from GitHub Pages) can
call via fetch(). CORS is open so the static site can reach it cross-origin.

Endpoints:
    GET  /          -> health check + model status
    POST /score     -> { "text": "..." } -> { discrepancy, probability, n_tokens, ok }

`probability` is a calibrated 0..1 AI-likelihood derived from the raw
discrepancy via a logistic squash. The squash constants (CENTER, SCALE) are
placeholders until eval_via_api.js fixes the operating threshold on real data —
see backend/README.md.
"""

import math

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from fast_detect import fast_detect_score, MODEL_NAME

# Logistic squash of raw discrepancy -> 0..1. Tune after eval_via_api.js.
CENTER = 1.0   # discrepancy value mapped to p=0.5
SCALE = 1.5    # spread; larger = softer transition

app = FastAPI(title="Fast-DetectGPT", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # static site is public; no secrets here
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ScoreIn(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/score")
def score(body: ScoreIn):
    r = fast_detect_score(body.text or "")
    d = r["discrepancy"]
    prob = 1.0 / (1.0 + math.exp(-(d - CENTER) / SCALE)) if r["ok"] else 0.5
    return {
        "discrepancy": d,
        "probability": prob,
        "n_tokens": r["n_tokens"],
        "ok": r["ok"],
    }
