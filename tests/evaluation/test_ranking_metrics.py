import pytest

from digsite.evaluation.ranking_metrics import (
    average_precision,
    ndcg_at,
    precision_at,
    recall_at,
    reciprocal_rank,
)

# "a" is highly relevant, "c" relevant, "x" relevant but never retrieved.
RANKING = ["a", "b", "c", "d", "e"]
GAINS = {"a": 3, "c": 2, "x": 1, "d": 0}


def test_precision_at_k() -> None:
    assert precision_at(RANKING, GAINS, 1) == 1.0
    assert precision_at(RANKING, GAINS, 2) == 0.5
    assert precision_at(RANKING, GAINS, 5) == pytest.approx(2 / 5)
    # A short ranking is not rewarded: the missing results count as misses.
    assert precision_at(RANKING, GAINS, 10) == pytest.approx(2 / 10)


def test_recall_at_k() -> None:
    assert recall_at(RANKING, GAINS, 1) == pytest.approx(1 / 3)
    assert recall_at(RANKING, GAINS, 5) == pytest.approx(2 / 3)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(RANKING, GAINS) == 1.0
    assert reciprocal_rank(["b", "d", "c"], GAINS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["b", "d"], GAINS) == 0.0


def test_average_precision() -> None:
    # Hits at ranks 1 and 3: (1/1 + 2/3) / 3 relevant items.
    assert average_precision(RANKING, GAINS) == pytest.approx((1 + 2 / 3) / 3)


def test_ndcg_at_k() -> None:
    # DCG = 3/log2(2) + 2/log2(4) = 4; ideal = 3 + 2/log2(3) + 1/log2(4).
    assert ndcg_at(RANKING, GAINS, 5) == pytest.approx(4 / (3 + 2 / 1.5849625 + 0.5))


def test_ndcg_rewards_putting_the_most_relevant_first() -> None:
    assert ndcg_at(["a", "c"], GAINS, 10) > ndcg_at(["c", "a"], GAINS, 10)


def test_a_perfect_ranking_scores_one() -> None:
    perfect = ["a", "c", "x"]

    assert ndcg_at(perfect, GAINS, 10) == pytest.approx(1.0)
    assert average_precision(perfect, GAINS) == pytest.approx(1.0)
    assert recall_at(perfect, GAINS, 3) == 1.0


def test_negative_gains_count_as_not_relevant() -> None:
    gains = {"a": 1, "b": -1}

    assert precision_at(["b", "a"], gains, 2) == 0.5
    assert ndcg_at(["b", "a"], gains, 2) == pytest.approx(1 / 1.5849625)


@pytest.mark.parametrize("gains", [{}, {"a": 0}])
def test_without_relevant_items_every_metric_is_zero(gains: dict[str, int]) -> None:
    assert precision_at(RANKING, gains, 5) == 0.0
    assert recall_at(RANKING, gains, 5) == 0.0
    assert reciprocal_rank(RANKING, gains) == 0.0
    assert average_precision(RANKING, gains) == 0.0
    assert ndcg_at(RANKING, gains, 5) == 0.0


def test_an_empty_ranking_scores_zero() -> None:
    assert precision_at([], GAINS, 5) == 0.0
    assert ndcg_at([], GAINS, 5) == 0.0
    assert precision_at(RANKING, GAINS, 0) == 0.0
