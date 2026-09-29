"""Typer CLI: run / diff / report / history."""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import SuiteSpec
from .diff import diff_runs
from .report import render_diff_html, render_run_html, write_html
from .runner import run_suite
from .store import Store


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from a local .env into os.environ (no dependency).

    Lets litellm find provider keys (e.g. GROQ_API_KEY) without the user
    exporting them per-shell. .env is git-ignored — never commit real keys.
    """
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


app = typer.Typer(add_completion=False, help="CI gate for LLM quality.")
console = Console()

STORE_OPT = typer.Option(".llm-gate/store.db", "--store", help="Path to the run store SQLite file.")


@app.callback()
def _startup() -> None:
    """Load .env (provider API keys) before any subcommand runs."""
    _load_dotenv()
    # price lookups from the bundled map: no network call, works offline/air-gapped
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@app.command()
def run(
    suite_path: str = typer.Argument(..., help="Path to suite.yaml"),
    notes: str = typer.Option(
        "", "--notes", help="Free-text note recorded with the run (e.g. git sha)."
    ),
    store: str = STORE_OPT,
    report_out: str = typer.Option("", "--report", help="Also write an HTML report to this path."),
) -> None:
    """Run a suite: dataset -> target -> metrics -> stored run."""
    suite = SuiteSpec.load(suite_path)
    record = run_suite(suite, store_path=store, notes=notes)

    t = Table(title=f"{record.suite} — run {record.run_id}")
    t.add_column("examples", justify="right")
    t.add_column("pass rate", justify="right")
    for m in record.mean_scores:
        t.add_column(m, justify="right")
    t.add_column("cost", justify="right")
    t.add_row(
        str(record.n_examples),
        f"{record.pass_rate:.0%}",
        *[f"{v:.3f}" for v in record.mean_scores.values()],
        f"${record.total_cost_usd:.4f}",
    )
    console.print(t)
    if report_out:
        path = write_html(render_run_html(record), report_out)
        console.print(f"report -> [cyan]{path}[/cyan]")


@app.command()
def diff(
    run_a: str = typer.Argument(..., help="Baseline run id (prefix ok)"),
    run_b: str = typer.Argument(..., help="New run id (prefix ok)"),
    store: str = STORE_OPT,
    report_out: str = typer.Option("", "--report", help="Write an HTML diff report to this path."),
    fail_on_regression: bool = typer.Option(
        False, "--fail-on-regression", help="Exit code 1 on any pass->fail flip (for CI)."
    ),
) -> None:
    """Compare two runs; the regression gate."""
    s = Store(store)
    a, b = s.get_run(run_a), s.get_run(run_b)
    s.close()
    if a is None or b is None:
        console.print(f"[red]run not found[/red] (a={run_a!r} b={run_b!r}); try `llm-gate history`")
        raise typer.Exit(2)

    report = diff_runs(a, b)

    t = Table(title=f"diff {a.run_id} -> {b.run_id} ({a.suite})")
    t.add_column("metric")
    t.add_column("run A", justify="right")
    t.add_column("run B", justify="right")
    t.add_column("delta mean", justify="right")
    t.add_column("95% CI")
    t.add_column("verdict")
    for d in report.metric_deltas:
        color = "green" if d.verdict == "improved" else ("red" if d.verdict == "regressed" else "")
        t.add_row(
            d.metric,
            f"{d.mean_a:.3f}",
            f"{d.mean_b:.3f}",
            f"{d.delta:+.3f}",
            escape(f"[{d.ci_low:+.3f}, {d.ci_high:+.3f}]"),
            f"[{color}]{d.verdict}[/{color}]" if color else d.verdict,
        )
    console.print(t)

    for f in report.regressions:
        console.print(f"[red]REGRESSION[/red] {f.example_id}")
    for f in report.fixes:
        console.print(f"[yellow]FIX[/yellow] {f.example_id}")
    if report.ok:
        console.print("[green bold]GATE PASSED[/green bold] — no pass->fail regressions")
    else:
        console.print(f"[red bold]GATE FAILED[/red bold] — {len(report.regressions)} regression(s)")

    if report_out:
        path = write_html(render_diff_html(report), report_out)
        console.print(f"report -> [cyan]{path}[/cyan]")
    if fail_on_regression and not report.ok:
        raise typer.Exit(1)


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run id (prefix ok)"),
    out: str = typer.Option("llm-gate-report.html", "-o", help="Output HTML path."),
    store: str = STORE_OPT,
) -> None:
    """Render a single run as a standalone HTML report."""
    s = Store(store)
    rec = s.get_run(run_id)
    s.close()
    if rec is None:
        console.print(f"[red]run {run_id!r} not found[/red]")
        raise typer.Exit(2)
    path = write_html(render_run_html(rec), out)
    console.print(f"report -> [cyan]{path}[/cyan]")


@app.command()
def history(
    limit: int = typer.Option(20, "--limit"),
    store: str = STORE_OPT,
) -> None:
    """List recent runs."""
    if not Path(store).exists():
        console.print("[yellow]no runs yet[/yellow]")
        return
    s = Store(store)
    rows = s.list_runs(limit)
    s.close()
    t = Table(title="runs")
    for col in ("run_id", "suite", "started_at", "n_examples", "pass_rate", "cost"):
        t.add_column(col)
    for r in rows:
        t.add_row(
            r["run_id"], r["suite"], r["started_at"], str(r["n_examples"]),
            f"{r['pass_rate']:.0%}" if r["pass_rate"] is not None else "-",
            f"${r['total_cost_usd']:.4f}",
        )
    console.print(t)


if __name__ == "__main__":
    app()
