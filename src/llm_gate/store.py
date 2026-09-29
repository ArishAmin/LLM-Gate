"""Run store — SQLite, one file at .llm-gate/store.db.

Two tables: runs (metadata) and results (one row per example per run).
Scores are stored as JSON so the schema survives adding new metrics.
`llm-gate history` queries this same file with DuckDB when installed.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    suite TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    n_examples INTEGER DEFAULT 0,
    pass_rate REAL,
    total_cost_usd REAL DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    mean_scores TEXT,
    notes TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    example_id TEXT NOT NULL,
    output TEXT,
    contexts TEXT,
    latency_ms REAL,
    scores TEXT NOT NULL,
    passed INTEGER NOT NULL,
    error TEXT,
    PRIMARY KEY (run_id, example_id),
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);
"""


@dataclass
class ExampleResult:
    example_id: str
    output: str
    contexts: list[str]
    latency_ms: float
    scores: dict[str, float]
    passed: bool
    error: str | None = None


@dataclass
class RunRecord:
    run_id: str
    suite: str
    started_at: str = ""
    finished_at: str | None = None
    n_examples: int = 0
    pass_rate: float | None = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    mean_scores: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    results: list[ExampleResult] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path = ".llm-gate/store.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    def new_run(self, suite_name: str, notes: str = "") -> str:
        run_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            "INSERT INTO runs (run_id, suite, started_at, notes) VALUES (?, ?, ?, ?)",
            (run_id, suite_name, _now(), notes),
        )
        self._conn.commit()
        return run_id

    def add_result(self, run_id: str, r: ExampleResult) -> None:
        self._conn.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                r.example_id,
                r.output,
                json.dumps(r.contexts),
                r.latency_ms,
                json.dumps(r.scores),
                int(r.passed),
                r.error,
            ),
        )
        self._conn.commit()

    def finish_run(self, run_id: str, record: RunRecord) -> None:
        self._conn.execute(
            """UPDATE runs SET finished_at=?, n_examples=?, pass_rate=?,
               total_cost_usd=?, total_tokens=?, mean_scores=? WHERE run_id=?""",
            (
                _now(),
                record.n_examples,
                record.pass_rate,
                record.total_cost_usd,
                record.total_tokens,
                json.dumps(record.mean_scores),
                run_id,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id LIKE ?", (run_id + "%",)
        ).fetchone()
        if row is None:
            return None
        rec = RunRecord(
            run_id=row["run_id"],
            suite=row["suite"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            n_examples=row["n_examples"] or 0,
            pass_rate=row["pass_rate"],
            total_cost_usd=row["total_cost_usd"] or 0.0,
            total_tokens=row["total_tokens"] or 0,
            mean_scores=json.loads(row["mean_scores"] or "{}"),
            notes=row["notes"] or "",
        )
        for r in self._conn.execute(
            "SELECT * FROM results WHERE run_id = ? ORDER BY example_id", (rec.run_id,)
        ):
            rec.results.append(
                ExampleResult(
                    example_id=r["example_id"],
                    output=r["output"] or "",
                    contexts=json.loads(r["contexts"] or "[]"),
                    latency_ms=r["latency_ms"] or 0.0,
                    scores=json.loads(r["scores"]),
                    passed=bool(r["passed"]),
                    error=r["error"],
                )
            )
        return rec

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT run_id, suite, started_at, n_examples, pass_rate, total_cost_usd "
            "FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
