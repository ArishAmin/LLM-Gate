"""Deterministic metrics — no LLM calls, fully reproducible, free.

These are the metrics you should maximize coverage with before paying for
judges: anything checkable without a model should be checked without one.
"""

from __future__ import annotations

import json
import re

from ..config import Example
from . import register

_WORD = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _token_list(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@register("exact_match")
def exact_match(example: Example, output: str, contexts: list[str], params: dict) -> float:
    if example.reference is None:
        raise ValueError("exact_match requires a reference")
    normalize = params.get("normalize_whitespace", True)
    a, b = output, example.reference
    if normalize:
        a, b = " ".join(a.split()), " ".join(b.split())
    return 1.0 if a.strip() == b.strip() else 0.0


@register("contains")
def contains(example: Example, output: str, contexts: list[str], params: dict) -> float:
    """All `params["all_of"]` substrings must appear (case-insensitive)."""
    needles = params.get("all_of", [])
    if not needles:
        raise ValueError("contains requires params.all_of (list of substrings)")
    low = output.lower()
    hits = sum(1 for n in needles if n.lower() in low)
    return hits / len(needles)


@register("f1")
def token_f1(example: Example, output: str, contexts: list[str], params: dict) -> float:
    """Token-level F1 vs the reference — SQuAD style."""
    if example.reference is None:
        raise ValueError("f1 requires a reference")
    pred, gold = _token_list(output), _token_list(example.reference)
    if not pred or not gold:
        return 0.0
    common: dict[str, int] = {}
    for t in gold:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in pred:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@register("context_recall")
def context_recall(example: Example, output: str, contexts: list[str], params: dict) -> float:
    """Fraction of reference tokens covered by the retrieved contexts.

    A retrieval-side signal: if the answer's facts aren't in the contexts,
    the generator can only be right by luck (or by hallucinating).
    """
    if example.reference is None:
        raise ValueError("context_recall requires a reference")
    ctx_tokens = _tokens(" ".join(contexts))
    ref = _tokens(example.reference)
    if not ref:
        return 1.0
    return len(ref & ctx_tokens) / len(ref)


@register("json_valid")
def json_valid(example: Example, output: str, contexts: list[str], params: dict) -> float:
    """1.0 when the output parses as JSON (optionally matching a key set)."""
    try:
        data = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    required = params.get("required_keys", [])
    if required and isinstance(data, dict):
        return 1.0 if all(k in data for k in required) else 0.0
    return 1.0
