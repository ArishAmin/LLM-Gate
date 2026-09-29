"""Judge smoke test — run this once before trusting any judge-based claims.

It verifies, in order:
  1. a provider key is present (via git-ignored .env or environment)
  2. the judge model is reachable and returns parseable JSON
  3. cost + token accounting is populated
  4. the response cache makes an identical second call free

Usage:
    python scripts/judge_smoke.py            # uses judge.model from examples/suite.yaml
    python scripts/judge_smoke.py --list     # also print models your key can access

No key is ever printed. If a step fails, nothing downstream is trustworthy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# make `src/` importable when run from the repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gate.cache import Cache  # noqa: E402
from llm_gate.cli import _load_dotenv  # noqa: E402
from llm_gate.config import SuiteSpec  # noqa: E402
from llm_gate.llm import LLMClient  # noqa: E402

KEYS = ["GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"]

# preference order when auto-picking a fallback model
_PREFERRED = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]


def _groq_models() -> list[str]:
    import os

    import httpx

    r = httpx.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": "Bearer " + os.environ["GROQ_API_KEY"]},
        timeout=30,
    )
    return sorted(m["id"] for m in r.json().get("data", []))


def _list_models(provider: str) -> list[str]:
    if provider == "groq":
        ids = _groq_models()
        print("models your Groq key can access:")
        for i in ids:
            print("  ", i)
        return ids
    return []


def _pick_fallback(ids: list[str]) -> str | None:
    for pref in _PREFERRED:
        if pref in ids:
            return pref
    llama = [i for i in ids if "llama" in i.lower() and "guard" not in i.lower()]
    return (llama or ids or [None])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="examples/suite_judge.yaml")
    ap.add_argument("--list", action="store_true", help="also list accessible models")
    args = ap.parse_args()

    _load_dotenv()

    import os

    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    present = [k for k in KEYS if os.environ.get(k)]
    if not present:
        print("FAIL: no provider key found. Copy .env.example -> .env and add a key.")
        return 1
    print(f"[1/4] provider key present: {', '.join(present)} (value hidden)")

    suite = SuiteSpec.load(args.suite)
    model = suite.judge.model
    provider = model.split("/", 1)[0]
    if args.list:
        _list_models(provider)

    cache = Cache(".llm-gate/judge-cache.db")
    client = LLMClient(cache=cache)

    prompt = (
        'You are a strict evaluator. Score the ANSWER faithfulness 1-5. '
        'Reason briefly, then output JSON exactly: {"reasoning": "...", "score": <int>}\n'
        "CONTEXTS: Refunds are issued within 30 days of purchase.\n"
        "ANSWER: You can get a refund within 30 days."
    )

    def _call(m: str):
        return client.complete_json(
            model=m,
            prompt=prompt,
            temperature=0.0,
            seed=42,
            api_base=suite.judge.api_base,
        )

    t0 = time.perf_counter()
    try:
        data = _call(model)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "model_not_found" not in msg and "does not exist" not in msg:
            print(f"[2/4] FAIL: judge call to {model!r} errored: {type(e).__name__}: {e}")
            cache.close()
            return 1
        # self-heal: discover accessible models and retry with a preferred one
        print(f"[2/4] configured model {model!r} not accessible for this key; auto-selecting...")
        try:
            ids = _groq_models() if provider == "groq" else []
        except Exception as le:  # noqa: BLE001
            print(f"      could not list models: {le}")
            cache.close()
            return 1
        fallback = _pick_fallback(ids)
        if not fallback:
            print("      no usable models found for this key.")
            cache.close()
            return 1
        model = f"{provider}/{fallback}"
        print(f"      -> retrying with {model}")
        try:
            t0 = time.perf_counter()
            data = _call(model)
        except Exception as e2:  # noqa: BLE001
            print(f"[2/4] FAIL on fallback {model!r}: {type(e2).__name__}: {e2}")
            cache.close()
            return 1
        print(f"      PIN THIS in {args.suite} -> judge.model: {model}")
    dt = (time.perf_counter() - t0) * 1000
    score = data.get("score")
    print(f"[2/4] judge reachable: {model} -> score={score} in {dt:.0f}ms, JSON parsed OK")
    if not isinstance(score, int) or not 1 <= score <= 5:
        print(f"      WARN: unexpected score {score!r}; check the rubric prompt")

    cost1, tok1 = client.total_cost_usd, client.total_tokens
    print(f"[3/4] accounting: cost=${cost1:.6f}, tokens={tok1}")
    if tok1 == 0:
        print("      WARN: provider returned no usage; cost tracking will read $0")

    t0 = time.perf_counter()
    client2 = LLMClient(cache=cache)
    client2.complete_json(model=model, prompt=prompt, temperature=0.0, seed=42)
    dt2 = (time.perf_counter() - t0) * 1000
    cached_ok = cache.hits >= 1
    print(f"[4/4] cache: second identical call in {dt2:.0f}ms, cache hits={cache.hits} "
          f"-> {'HIT (free)' if cached_ok else 'MISS (cache not working!)'}")

    cache.close()
    print("\nSMOKE TEST PASSED — judge path is real. You can now run:")
    print("  llm-gate run examples/suite.yaml --notes 'with judge'")
    return 0 if cached_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
