"""Provider compatibility tests using local stub servers (no network, no keys).

Two in-process HTTP stubs emulate the two wire formats llm-gate's judge speaks:
  - OpenAI-compatible  POST /v1/chat/completions   (OpenAI, Groq, Gemini-compat,
    LM Studio, vLLM, llama.cpp, LocalAI, OpenRouter, Azure...)
  - Ollama             POST /api/chat              (local Ollama daemon)

If these pass, any server speaking either protocol works as a judge target.
"""

from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

os.environ.setdefault("OPENAI_API_KEY", "sk-stub-not-real")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # no remote cost-map fetch

from llm_gate.cache import Cache  # noqa: E402
from llm_gate.config import Example, SuiteSpec  # noqa: E402
from llm_gate.llm import LLMClient  # noqa: E402
from llm_gate.metrics.judge import faithfulness  # noqa: E402
from llm_gate.runner import run_suite  # noqa: E402

JUDGE_JSON = '{"reasoning": "stub says supported", "score": 4}'


class _StubHandler(BaseHTTPRequestHandler):
    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802 — http.server API
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.endswith("/chat/completions"):
            self._send(
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "stub",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": JUDGE_JSON},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
                }
            )
        elif self.path.endswith("/api/chat"):
            self._send(
                {
                    "model": "stub",
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": JUDGE_JSON},
                    "done_reason": "stop",
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 7,
                }
            )
        else:
            self.send_error(404)

    def log_message(self, *args):  # silence request logging
        pass


@pytest.fixture(scope="module")
def stub_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_openai_compatible_endpoint(stub_server, tmp_path):
    client = LLMClient(cache=Cache(tmp_path / "c.db"))
    data = client.complete_json(
        model="openai/stub-model", prompt="score this", api_base=f"{stub_server}/v1"
    )
    assert data == {"reasoning": "stub says supported", "score": 4}
    assert client.total_tokens == 19  # usage parsed from OpenAI shape


def test_ollama_endpoint(stub_server, tmp_path):
    client = LLMClient(cache=Cache(tmp_path / "c.db"))
    data = client.complete_json(
        model="ollama_chat/stub", prompt="score this", api_base=stub_server
    )
    assert data["score"] == 4
    assert client.total_tokens == 19  # usage parsed from Ollama eval counts


def test_judge_metric_over_custom_endpoint(stub_server, tmp_path):
    client = LLMClient(cache=Cache(tmp_path / "c.db"))
    params = {
        "_client": client,
        "_judge": {
            "model": "openai/stub-model",
            "api_base": f"{stub_server}/v1",
            "samples": 2,
            "temperature": 0.0,
            "seed": 42,
        },
    }
    ex = Example(id="t", input="q", reference="r")
    score = faithfulness(ex, "answer", ["context"], params)
    assert abs(score - 0.75) < 1e-9  # score 4 -> (4-1)/4


def test_cache_key_includes_api_base(tmp_path):
    a = Cache.key("m", "p", 0.0, 1, "http://a")
    b = Cache.key("m", "p", 0.0, 1, "http://b")
    assert a != b  # same prompt on two servers must not share cache entries


def test_full_run_over_custom_endpoint(stub_server, tmp_path):
    """The whole pipeline (suite -> target -> judge metric -> store) on a local server."""
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "data.jsonl").write_text(
        '{"id": "e1", "input": "How long for refunds?", '
        '"reference": "Refunds are issued within 30 days of purchase."}\n',
        encoding="utf-8",
    )
    (examples / "target.py").write_text(
        "def answer(example):\n"
        "    return {'output': 'Refunds are issued within 30 days of purchase.', "
        "'contexts': ['Refunds are issued within 30 days of purchase.']}\n",
        encoding="utf-8",
    )
    suite = SuiteSpec.model_validate(
        {
            "name": "stub-suite",
            "dataset": {"path": "data.jsonl"},
            "target": {"callable": "target:answer"},
            "metrics": [{"name": "faithfulness", "threshold": 0.5}],
            "judge": {
                "model": "openai/stub-model",
                "api_base": f"{stub_server}/v1",
                "samples": 1,
            },
        }
    )
    suite._base_dir = examples
    record = run_suite(suite, store_path=tmp_path / "store.db")
    assert record.n_examples == 1
    assert record.pass_rate == 1.0
    assert abs(record.mean_scores["faithfulness"] - 0.75) < 1e-9
