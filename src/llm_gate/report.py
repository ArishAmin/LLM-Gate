"""Self-contained HTML report (Jinja2).

Deliberate v1 choice: a static HTML file, not a web dashboard. Zero JS build
chain, shareable by dropping it in Slack, and diff-able artifacts in CI.
A live dashboard is a v2 concern once the file format stabilizes.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .diff import DiffReport
from .store import RunRecord

_TEMPLATES = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        autoescape=select_autoescape(["html"]),
    )


def render_run_html(record: RunRecord) -> str:
    return _env().get_template("report.html.j2").render(run=record, diff=None)


def render_diff_html(report: DiffReport) -> str:
    return _env().get_template("report.html.j2").render(
        run=report.run_b, diff=report, run_a=report.run_a
    )


def write_html(html: str, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
