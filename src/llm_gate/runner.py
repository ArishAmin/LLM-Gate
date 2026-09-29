"""The eval runner: dataset -> target -> metrics -> pass/fail -> store."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console

from .cache import Cache
from .config import SuiteSpec
from .llm import LLMClient
from .metrics import JUDGE_METRICS, compute
from .store import ExampleResult, RunRecord, Store
from .targets import resolve_target

console = Console()


def _needs_judge(suite: SuiteSpec) -> bool:
    return any(m.name in JUDGE_METRICS for m in suite.metrics)


def run_suite(
    suite: SuiteSpec,
    store_path: str | Path = ".llm-gate/store.db",
    notes: str = "",
) -> RunRecord:
    examples = suite.load_dataset()
    target = resolve_target(suite)
    store = Store(store_path)

    client: LLMClient | None = None
    if _needs_judge(suite):
        cache = Cache(Path(store_path).parent / "judge-cache.db")
        client = LLMClient(cache=cache)

    run_id = store.new_run(suite.name, notes)
    record = RunRecord(
        run_id=run_id,
        suite=suite.name,
        started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )
    thresholds = {m.name: m.threshold for m in suite.metrics}

    console.print(f"[bold]llm-gate[/bold] run [cyan]{run_id}[/cyan] — suite '{suite.name}' "
                  f"({len(examples)} examples, {len(suite.metrics)} metrics)")

    for i, ex in enumerate(examples, 1):
        t0 = time.perf_counter()
        try:
            out = target(ex.model_dump())
            output = str(out.get("output", ""))
            contexts = list(out.get("contexts", ex.contexts))
            error = None
        except Exception:
            output, contexts = "", []
            error = traceback.format_exc(limit=3)
        latency_ms = (time.perf_counter() - t0) * 1000

        scores: dict[str, float] = {}
        for spec in suite.metrics:
            params = dict(spec.params)
            if spec.name in JUDGE_METRICS:
                if client is None:
                    raise RuntimeError(
                        f"metric {spec.name!r} needs an LLM judge — "
                        'install llm-gate[judge] and set your provider API key'
                    )
                params["_client"] = client
                params["_judge"] = suite.judge.model_dump()
            try:
                scores[spec.name] = compute(spec.name, ex, output, contexts, params)
            except Exception as e:
                scores[spec.name] = 0.0
                error = (error or "") + f"\nmetric {spec.name}: {e}"

        passed = error is None and all(
            scores.get(name, 0.0) >= thr for name, thr in thresholds.items()
        )
        result = ExampleResult(
            example_id=ex.id,
            output=output,
            contexts=contexts,
            latency_ms=latency_ms,
            scores=scores,
            passed=passed,
            error=error,
        )
        store.add_result(run_id, result)
        record.results.append(result)
        mark = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        console.print(f"  [{i}/{len(examples)}] {mark} {ex.id}")

    n = len(record.results)
    record.n_examples = n
    record.pass_rate = sum(r.passed for r in record.results) / n if n else 0.0
    record.total_cost_usd = client.total_cost_usd if client else 0.0
    record.total_tokens = client.total_tokens if client else 0
    for spec in suite.metrics:
        vals = [r.scores[spec.name] for r in record.results if spec.name in r.scores]
        record.mean_scores[spec.name] = sum(vals) / len(vals) if vals else 0.0

    store.finish_run(run_id, record)
    store.close()
    if client and client.cache:
        client.cache.close()
    return record
