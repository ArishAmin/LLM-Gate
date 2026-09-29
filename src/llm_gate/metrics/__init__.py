"""Metric registry.

A metric is fn(example: Example, output: str, contexts: list[str], params: dict) -> float
returning a score in [0, 1]. Deterministic metrics live in deterministic.py,
LLM-as-judge metrics in judge.py; both self-register here.
"""

from __future__ import annotations

from collections.abc import Callable

from ..config import Example

MetricFn = Callable[..., float]

METRICS: dict[str, MetricFn] = {}


def register(name: str) -> Callable[[MetricFn], MetricFn]:
    def deco(fn: MetricFn) -> MetricFn:
        if name in METRICS:
            raise ValueError(f"metric {name!r} registered twice")
        METRICS[name] = fn
        return fn

    return deco


def compute(name: str, example: Example, output: str, contexts: list[str], params: dict) -> float:
    if name not in METRICS:
        raise KeyError(f"unknown metric {name!r}; available: {sorted(METRICS)}")
    score = METRICS[name](example=example, output=output, contexts=contexts, params=params)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"metric {name!r} returned {score}; scores must be in [0, 1]")
    return score


# importing submodules populates the registry
from . import deterministic, judge  # noqa: E402,F401
from .judge import JUDGE_METRICS as JUDGE_METRICS  # noqa: E402
