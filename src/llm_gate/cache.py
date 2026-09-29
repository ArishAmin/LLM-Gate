"""Content-hash keyed response cache (SQLite).

Why: judge calls cost money and time; re-running a suite whose inputs haven't
changed should be free and instant. The key hashes everything that affects the
response (model, prompt, temperature, seed), so any change invalidates cleanly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


class Cache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(value, default=str)),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
