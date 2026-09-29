from pathlib import Path

from llm_gate.config import SuiteSpec

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_load_suite():
    suite = SuiteSpec.load(EXAMPLES / "suite.yaml")
    assert suite.name == "demo-support-rag"
    assert {m.name for m in suite.metrics} == {"f1", "context_recall"}
    assert suite.judge.samples == 2


def test_judge_suite_declares_judge_metrics():
    suite = SuiteSpec.load(EXAMPLES / "suite_judge.yaml")
    names = {m.name for m in suite.metrics}
    assert {"faithfulness", "answer_relevance"} <= names
    assert suite.judge.model.startswith("groq/")


def test_load_dataset():
    suite = SuiteSpec.load(EXAMPLES / "suite.yaml")
    rows = suite.load_dataset()
    assert len(rows) == 10
    assert rows[0].id == "q-refund-window"
    assert all(r.reference for r in rows)


def test_relative_paths_resolve_from_suite_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # suite must load from any cwd
    suite = SuiteSpec.load(EXAMPLES / "suite.yaml")
    assert suite.dataset_path().exists()
