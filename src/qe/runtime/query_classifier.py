"""Lightweight query classifier for adaptive memory tier weighting.

Classifies user queries into types (factual, procedural, analytical,
creative, meta) using regex + keyword matching, then provides memory
tier weights appropriate for each query type.
"""

from __future__ import annotations

import re

# Query type → memory tier weights
# Tiers: episodic (past interactions), semantic (claims/knowledge),
#         procedural (how-to), working (current session)
QUERY_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "factual": {
        "semantic": 0.50,
        "episodic": 0.15,
        "procedural": 0.10,
        "working": 0.25,
    },
    "procedural": {
        "semantic": 0.10,
        "episodic": 0.20,
        "procedural": 0.50,
        "working": 0.20,
    },
    "analytical": {
        "semantic": 0.35,
        "episodic": 0.20,
        "procedural": 0.10,
        "working": 0.35,
    },
    "creative": {
        "semantic": 0.20,
        "episodic": 0.30,
        "procedural": 0.05,
        "working": 0.45,
    },
    "meta": {
        "semantic": 0.10,
        "episodic": 0.40,
        "procedural": 0.20,
        "working": 0.30,
    },
    "conversational": {
        "semantic": 0.05,
        "episodic": 0.15,
        "procedural": 0.05,
        "working": 0.75,
    },
}

# Classification patterns
_FACTUAL_PATTERNS = [
    re.compile(r"\b(what|who|where|when|which)\b.*\?", re.I),
    re.compile(r"\b(tell me about|define|explain what)\b", re.I),
    re.compile(r"\b(is it true|fact|claim)\b", re.I),
]

_PROCEDURAL_PATTERNS = [
    re.compile(r"\b(how (do|can|to|should)|steps to|guide)\b", re.I),
    re.compile(r"\b(tutorial|walkthrough|instructions)\b", re.I),
    re.compile(r"\b(set up|install|configure|implement)\b", re.I),
]

_ANALYTICAL_PATTERNS = [
    re.compile(r"\b(why|analyze|compare|evaluate|assess)\b", re.I),
    re.compile(r"\b(trade-?offs?|pros and cons|implications)\b", re.I),
    re.compile(r"\b(difference between|relationship between)\b", re.I),
]

_CREATIVE_PATTERNS = [
    re.compile(r"\b(create|generate|write|compose|design|brainstorm)\b", re.I),
    re.compile(r"\b(idea|suggest|propose|imagine)\b", re.I),
]

_META_PATTERNS = [
    re.compile(r"\b(what do you (know|think|remember))\b", re.I),
    re.compile(r"\b(your (memory|knowledge|capabilities))\b", re.I),
    re.compile(r"\b(previous (conversation|session|discussion))\b", re.I),
]

_CONVERSATIONAL_PATTERNS = [
    re.compile(r"^(hi|hello|hey|thanks|ok|yes|no|sure|great)\b", re.I),
    re.compile(r"^(good (morning|afternoon|evening))\b", re.I),
]


def classify_query(query: str) -> str:
    """Classify a query into a type using regex + keyword matching.

    Returns one of: factual, procedural, analytical, creative, meta, conversational
    """
    scores: dict[str, int] = {
        "factual": 0,
        "procedural": 0,
        "analytical": 0,
        "creative": 0,
        "meta": 0,
        "conversational": 0,
    }

    for pat in _CONVERSATIONAL_PATTERNS:
        if pat.search(query):
            scores["conversational"] += 3

    for pat in _FACTUAL_PATTERNS:
        if pat.search(query):
            scores["factual"] += 2

    for pat in _PROCEDURAL_PATTERNS:
        if pat.search(query):
            scores["procedural"] += 2

    for pat in _ANALYTICAL_PATTERNS:
        if pat.search(query):
            scores["analytical"] += 2

    for pat in _CREATIVE_PATTERNS:
        if pat.search(query):
            scores["creative"] += 2

    for pat in _META_PATTERNS:
        if pat.search(query):
            scores["meta"] += 2

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best] == 0:
        return "factual"  # default
    return best


def get_memory_weights(query: str) -> dict[str, float]:
    """Classify query and return memory tier weights."""
    query_type = classify_query(query)
    return dict(QUERY_TYPE_WEIGHTS[query_type])


def get_memory_weights_for_type(query_type: str) -> dict[str, float]:
    """Return memory tier weights for a given query type."""
    return dict(QUERY_TYPE_WEIGHTS.get(query_type, QUERY_TYPE_WEIGHTS["factual"]))
