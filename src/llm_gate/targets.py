"""Target resolution: import the user's pipeline callable from "module:function".

The suite file's directory is added to sys.path, so examples/ can ship a
demo target next to the suite YAML.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Protocol

from .config import SuiteSpec


class Target(Protocol):
    """A target receives an example dict and returns at least {"output": str}.

    Optional keys the runner records when present:
      contexts: list[str]  — what the pipeline actually retrieved
      usage: {"prompt_tokens": int, "completion_tokens": int}
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]: ...


def resolve_target(suite: SuiteSpec) -> Target:
    spec = suite.target.callable
    if ":" not in spec:
        raise ValueError(f"target.callable must be 'module:function', got {spec!r}")
    module_name, func_name = spec.split(":", 1)
    base = str(suite.base_dir)
    if base not in sys.path:
        sys.path.insert(0, base)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} did not resolve to a callable")
    return fn  # type: ignore[return-value]
