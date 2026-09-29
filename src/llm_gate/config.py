"""Suite configuration models and dataset loading.

A suite YAML fully describes an eval run: which dataset, which target
pipeline, which metrics (with pass thresholds), and judge settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, PrivateAttr


class Example(BaseModel):
    """One dataset row. `contexts` are the chunks a RAG pipeline retrieved."""

    id: str
    input: str
    reference: str | None = None
    contexts: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class DatasetSpec(BaseModel):
    path: str  # JSONL file, one Example-shaped object per line


class TargetSpec(BaseModel):
    kind: Literal["python"] = "python"
    callable: str  # "module:function" — resolved relative to the suite file


class MetricSpec(BaseModel):
    name: str
    threshold: float = 0.0  # example passes when every metric score >= threshold
    params: dict[str, Any] = Field(default_factory=dict)


class JudgeSpec(BaseModel):
    model: str = "groq/openai/gpt-oss-120b"
    temperature: float = 0.0
    samples: int = 1  # judge N times per example; report mean + std (judges are noisy)
    seed: int | None = 42
    # For local / self-hosted judges (Ollama, LM Studio, vLLM, llama.cpp, LocalAI):
    #   model: ollama_chat/llama3.2        api_base: http://localhost:11434
    #   model: openai/my-local-model       api_base: http://localhost:1234/v1
    api_base: str | None = None


class SuiteSpec(BaseModel):
    name: str
    dataset: DatasetSpec
    target: TargetSpec
    metrics: list[MetricSpec]
    judge: JudgeSpec = Field(default_factory=JudgeSpec)

    # resolved dir of the suite file, so relative dataset/target paths work from any cwd
    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @classmethod
    def load(cls, path: str | Path) -> SuiteSpec:
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        suite = cls.model_validate(raw)
        suite._base_dir = path.parent.resolve()
        return suite

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def dataset_path(self) -> Path:
        p = Path(self.dataset.path)
        return p if p.is_absolute() else self.base_dir / p

    def load_dataset(self) -> list[Example]:
        rows: list[Example] = []
        with self.dataset_path().open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                data.setdefault("id", f"ex-{i:04d}")
                rows.append(Example.model_validate(data))
        if not rows:
            raise ValueError(f"dataset {self.dataset_path()} is empty")
        return rows
