"""Judge metric logic, tested with a fake client (no API key, no network).

This covers the parts that are pure code — rubric normalization, multi-sample
averaging, and that the right fields reach the prompt — so a bug there can't
hide behind 'it needs a key to test'.
"""

from llm_gate.config import Example
from llm_gate.metrics.judge import answer_relevance, faithfulness


class FakeClient:
    """Returns a scripted sequence of judge scores, records prompts it received."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.prompts = []

    def complete_json(self, model, prompt, temperature, seed, api_base=None):
        self.prompts.append(prompt)
        return {"reasoning": "fake", "score": self._scores.pop(0)}


def _ex(input="How long for a refund?", reference=None):
    return Example(id="t", input=input, reference=reference)


def test_faithfulness_normalizes_and_averages_samples():
    # judge scores 5 and 3 -> (5-1)/4=1.0 and (3-1)/4=0.5 -> mean 0.75
    client = FakeClient([5, 3])
    params = {"_client": client, "_judge": {"model": "groq/x", "samples": 2, "temperature": 0.0}}
    score = faithfulness(
        _ex(), "You can get a refund within 30 days.", ["Refunds within 30 days."], params
    )
    assert abs(score - 0.75) < 1e-9
    assert len(client.prompts) == 2  # one call per sample


def test_answer_relevance_clamps_out_of_range_scores():
    client = FakeClient([9, 0])  # both out of 1..5 -> clamped to 5 and 1
    params = {"_client": client, "_judge": {"model": "groq/x", "samples": 2}}
    score = answer_relevance(_ex(), "answer", [], params)
    # clamp(9)->5 ->1.0 ; clamp(0)->1 ->0.0 ; mean 0.5
    assert abs(score - 0.5) < 1e-9


def test_judge_prompts_contain_the_right_fields():
    client = FakeClient([4])
    params = {"_client": client, "_judge": {"samples": 1}}
    faithfulness(_ex(), "ANSWER_TEXT", ["CONTEXT_TEXT"], params)
    assert "ANSWER_TEXT" in client.prompts[0]
    assert "CONTEXT_TEXT" in client.prompts[0]

    client2 = FakeClient([4])
    params2 = {"_client": client2, "_judge": {"samples": 1}}
    answer_relevance(_ex(input="QUESTION_TEXT"), "ANSWER_TEXT", [], params2)
    assert "QUESTION_TEXT" in client2.prompts[0]
    assert "ANSWER_TEXT" in client2.prompts[0]


def test_single_sample_default():
    client = FakeClient([5])
    params = {"_client": client, "_judge": {}}  # no samples key -> defaults to 1
    score = faithfulness(_ex(), "a", ["b"], params)
    assert score == 1.0
    assert len(client.prompts) == 1
