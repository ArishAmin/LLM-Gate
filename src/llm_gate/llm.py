"""Provider-agnostic LLM client with caching and cost accounting.

litellm is a lazy import (the `judge` extra) so the deterministic core of
llm-gate runs with zero ML dependencies — important for CI and low-spec machines.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import Cache


@dataclass
class LLMResponse:
    text: str
    data: dict[str, Any] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False


@dataclass
class LLMClient:
    """Thin wrapper: caching + retries + running cost/token totals."""

    cache: Cache | None = None
    max_retries: int = 2
    total_cost_usd: float = field(default=0.0, init=False)
    total_tokens: int = field(default=0, init=False)

    def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        seed: int | None = None,
        api_base: str | None = None,
    ) -> LLMResponse:
        key = (
            Cache.key(model, prompt, temperature, seed, api_base) if self.cache else None
        )
        if key is not None and self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                resp = LLMResponse(**hit, cached=True)
                self._tally(resp)
                return resp

        import litellm  # lazy: only needed for judge metrics

        kwargs: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        if api_base:
            # local / self-hosted servers: Ollama, LM Studio, vLLM, llama.cpp,
            # LocalAI — anything speaking OpenAI or Ollama HTTP API
            kwargs["api_base"] = api_base

        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = litellm.completion(**kwargs)
                break
            except Exception as e:  # noqa: BLE001 — retry any transient provider error
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(2**attempt)  # backoff: rate limits are the common case
        else:
            raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {last_err}")

        text = raw.choices[0].message.content or ""
        usage = getattr(raw, "usage", None)
        try:
            cost = float(litellm.completion_cost(completion_response=raw))
        except Exception:  # unknown model pricing — track tokens, skip cost
            cost = 0.0
        resp = LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=cost,
        )
        self._tally(resp)
        if key is not None and self.cache is not None:
            self.cache.put(
                key,
                {
                    "text": resp.text,
                    "data": resp.data,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        return resp

    def complete_json(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        seed: int | None = None,
        api_base: str | None = None,
    ) -> dict[str, Any]:
        """complete() + parse a JSON object out of the response (tolerates fences)."""
        resp = self.complete(model, prompt, temperature, seed, api_base)
        data = _extract_json(resp.text)
        if data is None:
            raise ValueError(f"judge response was not valid JSON: {resp.text[:200]!r}")
        resp.data = data
        return data

    def _tally(self, resp: LLMResponse) -> None:
        self.total_cost_usd += resp.cost_usd
        self.total_tokens += resp.prompt_tokens + resp.completion_tokens


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_BLOCK.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
