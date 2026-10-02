import pytest

from digsite.evaluation.metrics import SetMetrics, compare_sets


def test_counts_hits_misses_and_false_alarms() -> None:
    metrics = compare_sets(truth={"a", "b", "c", "d"}, predicted={"a", "b", "x"})

    assert metrics == SetMetrics(true_positives=2, false_positives=1, false_negatives=2)
    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(4 / 7)


def test_a_perfect_prediction_scores_one() -> None:
    metrics = compare_sets(truth={1, 2}, predicted={1, 2})

    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)


def test_scores_are_zero_rather_than_undefined() -> None:
    nothing_predicted = compare_sets(truth={1, 2}, predicted=set())
    nothing_to_find = compare_sets(truth=set(), predicted={1})
    nothing_at_all = compare_sets(truth=set[int](), predicted=set[int]())

    for metrics in (nothing_predicted, nothing_to_find, nothing_at_all):
        assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
