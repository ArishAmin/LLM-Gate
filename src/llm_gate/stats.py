"""Statistics for run comparison.

A 2% mean-score bump is meaningless if per-example noise is larger than that.
Paired bootstrap over per-example deltas gives a confidence interval on the
true mean difference — the honest way to claim "v2 is better".
"""

from __future__ import annotations

import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def paired_bootstrap_ci(
    deltas: list[float],
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI for the mean of paired deltas (run B - run A per example).

    Returns (mean_delta, ci_low, ci_high). Resamples examples with replacement;
    no normality assumptions, works on tiny datasets.
    """
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    point = mean(deltas)
    if n == 1:
        return point, point, point
    boots = []
    for _ in range(iters):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo = boots[max(0, int((alpha / 2) * iters) - 1)]
    hi = boots[min(iters - 1, int((1 - alpha / 2) * iters))]
    return point, lo, hi


def significant(point: float, lo: float, hi: float) -> str:
    """Interpret a CI: does it exclude zero?"""
    if lo > 0:
        return "improved"
    if hi < 0:
        return "regressed"
    return "no significant change"
