"""End-to-end: run both demo suites offline, diff them, verify the gate catches
the degraded v2 target. This is the same flow the CI gate uses."""

from pathlib import Path

import pytest

from llm_gate.config import SuiteSpec
from llm_gate.diff import diff_runs
from llm_gate.runner import run_suite

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    store = tmp_path_factory.mktemp("store") / "store.db"
    a = run_suite(SuiteSpec.load(EXAMPLES / "suite.yaml"), store_path=store)
    b = run_suite(SuiteSpec.load(EXAMPLES / "suite_v2_degraded.yaml"), store_path=store)
    return a, b


def test_good_target_passes(runs):
    a, _ = runs
    assert a.n_examples == 10
    assert a.pass_rate == 1.0
    assert a.total_cost_usd == 0.0  # offline: no judge calls


def test_degraded_target_regresses(runs):
    _, b = runs
    assert b.pass_rate < 1.0  # retrieval broken -> several examples fail


def test_diff_gate_fails_on_regression(runs):
    a, b = runs
    report = diff_runs(a, b)
    assert not report.ok
    regressed_ids = {f.example_id for f in report.regressions}
    # refund questions still answerable from the first doc; others must regress
    assert regressed_ids
    assert all(not eid.startswith("q-refund") for eid in regressed_ids)


def test_diff_same_run_is_clean(runs):
    a, _ = runs
    report = diff_runs(a, a)
    assert report.ok
    assert not report.flips
    assert all(d.delta == 0.0 for d in report.metric_deltas)


def test_diff_rejects_cross_suite(runs, tmp_path):
    a, _ = runs
    spec = SuiteSpec.model_validate(
        {
            "name": "different-suite",
            "dataset": {"path": "data/sample.jsonl"},
            "target": {"callable": "demo_target:answer"},
            "metrics": [{"name": "f1", "threshold": 0.25}],
        }
    )
    spec._base_dir = EXAMPLES
    other = run_suite(spec, store_path=tmp_path / "other.db")
    with pytest.raises(ValueError, match="different suites"):
        diff_runs(a, other)
