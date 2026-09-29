from llm_gate.stats import mean, paired_bootstrap_ci, significant, stdev


def test_mean_stdev():
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
    assert stdev([5.0]) == 0.0
    assert abs(stdev([1.0, 2.0, 3.0]) - 1.0) < 1e-9


def test_bootstrap_constant_deltas_degenerate_ci():
    point, lo, hi = paired_bootstrap_ci([0.5] * 10, iters=200)
    assert point == lo == hi == 0.5


def test_bootstrap_zero_centered_is_not_significant():
    deltas = [0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02] * 3
    point, lo, hi = paired_bootstrap_ci(deltas, iters=500)
    assert significant(point, lo, hi) == "no significant change"


def test_bootstrap_clear_improvement():
    deltas = [0.4, 0.5, 0.45, 0.6, 0.55, 0.42]
    point, lo, hi = paired_bootstrap_ci(deltas, iters=500)
    assert point > 0.4
    assert significant(point, lo, hi) == "improved"


def test_single_example_ci():
    assert paired_bootstrap_ci([0.3], iters=10) == (0.3, 0.3, 0.3)
