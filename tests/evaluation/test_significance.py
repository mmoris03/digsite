import pytest

from digsite.evaluation.significance import paired_randomization_test


def test_a_consistent_difference_is_significant() -> None:
    better = [0.9, 0.8, 0.85, 0.7, 0.95, 0.75, 0.8, 0.9, 0.85, 0.7, 0.8, 0.9]
    worse = [score - 0.2 for score in better]

    assert paired_randomization_test(better, worse) < 0.01


def test_differences_that_cancel_out_are_not_significant() -> None:
    first = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
    second = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]

    assert paired_randomization_test(first, second) == pytest.approx(1.0)


def test_identical_systems_have_nothing_to_tell_apart() -> None:
    scores = [0.5, 0.25, 1.0]

    assert paired_randomization_test(scores, scores) == 1.0
    assert paired_randomization_test([], []) == 1.0


def test_the_test_is_two_sided() -> None:
    first = [0.9, 0.8, 0.7, 0.9, 0.6, 0.8, 0.75, 0.9]
    second = [0.5, 0.6, 0.4, 0.5, 0.5, 0.3, 0.45, 0.6]

    assert paired_randomization_test(first, second) == paired_randomization_test(second, first)


def test_few_queries_cannot_give_a_small_p_value() -> None:
    # Three queries allow eight assignments; two of them are as extreme as the observed one.
    p_value = paired_randomization_test([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], trials=20_000)

    assert p_value == pytest.approx(0.25, abs=0.02)


def test_the_p_value_is_never_zero() -> None:
    first = [1.0] * 40
    second = [0.0] * 40

    assert paired_randomization_test(first, second, trials=100) == pytest.approx(1 / 101)


def test_the_same_seed_gives_the_same_result() -> None:
    first = [0.4, 0.9, 0.3, 0.8, 0.6, 0.55]
    second = [0.5, 0.6, 0.35, 0.5, 0.7, 0.4]

    assert paired_randomization_test(first, second, seed=7) == paired_randomization_test(
        first, second, seed=7
    )


def test_both_systems_need_the_same_queries() -> None:
    with pytest.raises(ValueError, match="same queries"):
        paired_randomization_test([0.5, 0.5], [0.5])
