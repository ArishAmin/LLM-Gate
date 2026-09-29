"""Run diffing — the regression gate.

Compares two runs example-by-example and reports:
- flips: pass->fail (regressions) and fail->pass (fixes)
- per-metric mean delta with a paired bootstrap CI
- latency / cost deltas

`gate()` returns True when there are zero pass->fail flips — that's the
function CI calls; wire it to `llm-gate diff --fail-on-regression`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import mean, paired_bootstrap_ci, significant
from .store import RunRecord


@dataclass
class Flip:
    example_id: str
    kind: str  # "regression" | "fix"
    output_a: str
    output_b: str


@dataclass
class MetricDelta:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str


@dataclass
class DiffReport:
    run_a: RunRecord
    run_b: RunRecord
    flips: list[Flip] = field(default_factory=list)
    metric_deltas: list[MetricDelta] = field(default_factory=list)
    latency_delta_ms: float = 0.0
    cost_delta_usd: float = 0.0

    @property
    def regressions(self) -> list[Flip]:
        return [f for f in self.flips if f.kind == "regression"]

    @property
    def fixes(self) -> list[Flip]:
        return [f for f in self.flips if f.kind == "fix"]

    @property
    def ok(self) -> bool:
        return not self.regressions


def diff_runs(a: RunRecord, b: RunRecord) -> DiffReport:
    if a.suite != b.suite:
        raise ValueError(f"runs are from different suites: {a.suite!r} vs {b.suite!r}")
    idx_a = {r.example_id: r for r in a.results}
    idx_b = {r.example_id: r for r in b.results}
    common = sorted(set(idx_a) & set(idx_b))

    report = DiffReport(run_a=a, run_b=b)

    for eid in common:
        ra, rb = idx_a[eid], idx_b[eid]
        if ra.passed and not rb.passed:
            report.flips.append(Flip(eid, "regression", ra.output, rb.output))
        elif not ra.passed and rb.passed:
            report.flips.append(Flip(eid, "fix", ra.output, rb.output))

    metrics = sorted({m for r in common for m in idx_a[r].scores} &
                     {m for r in common for m in idx_b[r].scores})
    for m in metrics:
        deltas = [idx_b[e].scores[m] - idx_a[e].scores[m] for e in common]
        point, lo, hi = paired_bootstrap_ci(deltas)
        report.metric_deltas.append(
            MetricDelta(
                metric=m,
                mean_a=mean([idx_a[e].scores[m] for e in common]),
                mean_b=mean([idx_b[e].scores[m] for e in common]),
                delta=point,
                ci_low=lo,
                ci_high=hi,
                verdict=significant(point, lo, hi),
            )
        )

    if common:
        report.latency_delta_ms = mean([idx_b[e].latency_ms for e in common]) - mean(
            [idx_a[e].latency_ms for e in common]
        )
    report.cost_delta_usd = b.total_cost_usd - a.total_cost_usd
    return report
