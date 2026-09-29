from llm_gate.config import Example
from llm_gate.metrics.deterministic import contains, context_recall, exact_match, token_f1


def _ex(reference: str, input: str = "q") -> Example:
    return Example(id="t", input=input, reference=reference)


def test_exact_match_normalizes_whitespace():
    ex = _ex("Refunds are issued within 30 days.")
    assert exact_match(ex, "  Refunds are issued\nwithin 30 days. ", [], {}) == 1.0
    assert exact_match(ex, "No.", [], {}) == 0.0


def test_contains_partial_credit():
    ex = _ex("")
    score = contains(ex, "Refunds take 30 days via the dashboard.", [],
                     {"all_of": ["30 days", "dashboard", "missing"]})
    assert abs(score - 2 / 3) < 1e-9


def test_token_f1_bounds():
    ex = _ex("the refund window is 30 days")
    assert token_f1(ex, "the refund window is 30 days", [], {}) == 1.0
    assert token_f1(ex, "completely unrelated answer text", [], {}) == 0.0
    partial = token_f1(ex, "refund window", [], {})
    assert 0.0 < partial < 1.0


def test_context_recall():
    ex = _ex("Refunds are issued within 30 days of purchase.")
    good = ["Refunds are issued within 30 days of purchase. Items must be unopened."]
    assert context_recall(ex, "", good, {}) == 1.0
    bad = ["Shipping takes 3 to 5 business days."]
    assert context_recall(ex, "", bad, {}) < 0.3
