"""Offline demo target: a tiny keyword-retrieval RAG pipeline over a 4-doc corpus.

No API keys, no network — `llm-gate run examples/suite.yaml` works out of the box.

Two targets ship here so you can demo the regression gate end-to-end:
  answer             — sane retrieval (top-k by keyword overlap)
  answer_v2_degraded — a "well-intentioned refactor" that broke retrieval
                       (only ever reads the first doc). Run both suites,
                       then `llm-gate diff` shows exactly what regressed.
"""

from __future__ import annotations

import re
from typing import Any

CORPUS = {
    "refund-policy": (
        "Refunds are issued within 30 days of purchase. "
        "Items must be unopened and include original packaging. "
        "Refund requests are submitted through the account dashboard."
    ),
    "shipping": (
        "Standard shipping takes 3 to 5 business days within the country. "
        "Express shipping takes 1 to 2 business days and costs extra. "
        "International shipping is available to 40 countries."
    ),
    "warranty": (
        "All hardware carries a 2 year warranty covering manufacturing defects. "
        "The warranty does not cover water damage or accidental drops. "
        "Warranty claims require the original order number."
    ),
    "contact": (
        "Support is available via email at help@example.com, 24/7. "
        "Phone support runs on weekdays from 9am to 6pm. "
        "Priority support is included for enterprise accounts."
    ),
}

_WORD = re.compile(r"\w+")

# Without this, "are"/"the" overlap ties the wrong doc to the top and dict
# order silently decides retrieval — exactly the kind of bug evals exist to catch.
STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "does", "i", "to", "of", "in", "for",
    "how", "what", "when", "where", "do", "by", "at", "on", "it", "and", "or",
}


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _overlap(q: set[str], text: str) -> int:
    """Naive morphological matching: 'ship' hits 'shipping', 'international'
    hits 'internationally'. A real pipeline uses embeddings — this is a stub."""
    doc_tokens = _tokens(text) - STOPWORDS
    return sum(1 for qt in q if any(qt in dt or dt in qt for dt in doc_tokens))


def _retrieve(query: str, k: int = 2, docs: dict[str, str] | None = None) -> list[str]:
    docs = docs or CORPUS
    q = _tokens(query) - STOPWORDS
    scored = sorted(docs.items(), key=lambda kv: _overlap(q, kv[1]), reverse=True)
    return [text for _, text in scored[:k]]


def _synthesize(query: str, contexts: list[str]) -> str:
    """Answer by picking corpus sentences that overlap the query (extractive stub)."""
    q = _tokens(query) - STOPWORDS
    hits = []
    for ctx in contexts:
        for sent in re.split(r"(?<=[.])\s+", ctx):
            if _overlap(q, sent) > 0:
                hits.append(sent.strip())
    hits.sort(key=lambda s: _overlap(q, s), reverse=True)
    return " ".join(hits[:2]) or (contexts[0] if contexts else "")


def answer(example: dict[str, Any]) -> dict[str, Any]:
    contexts = _retrieve(example["input"], k=2)
    return {"output": _synthesize(example["input"], contexts), "contexts": contexts}


def answer_v2_degraded(example: dict[str, Any]) -> dict[str, Any]:
    # The classic "temporary simplification" that ships: retrieval is skipped
    # and only the first document is ever consulted.
    contexts = list(CORPUS.values())[:1]
    return {"output": _synthesize(example["input"], contexts), "contexts": contexts}
