"""Metrics for ranked results.

Every function takes the ranking as a sequence of item ids, best first, and the
judgements as a mapping from item id to gain: how relevant the item is to the
query, with 0 or a missing id meaning not relevant.
"""

import math
from collections.abc import Mapping, Sequence


def precision_at[T](ranking: Sequence[T], gains: Mapping[T, int], k: int) -> float:
    """Fraction of the first `k` results that are relevant."""
    if k <= 0:
        return 0.0
    return sum(gains.get(item, 0) > 0 for item in ranking[:k]) / k


def recall_at[T](ranking: Sequence[T], gains: Mapping[T, int], k: int) -> float:
    """Fraction of the relevant items that appear among the first `k` results."""
    relevant = sum(gain > 0 for gain in gains.values())
    if relevant == 0:
        return 0.0
    return sum(gains.get(item, 0) > 0 for item in ranking[:k]) / relevant


def reciprocal_rank[T](ranking: Sequence[T], gains: Mapping[T, int]) -> float:
    """One over the rank of the first relevant result; 0 if there is none."""
    for rank, item in enumerate(ranking, start=1):
        if gains.get(item, 0) > 0:
            return 1.0 / rank
    return 0.0


def average_precision[T](ranking: Sequence[T], gains: Mapping[T, int]) -> float:
    """Mean of the precision at the rank of each relevant result.

    Relevant items that were not retrieved count as precision 0, so the measure
    rewards both finding the relevant items and ranking them early.
    """
    relevant = sum(gain > 0 for gain in gains.values())
    if relevant == 0:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranking, start=1):
        if gains.get(item, 0) > 0:
            hits += 1
            total += hits / rank
    return total / relevant


def ndcg_at[T](ranking: Sequence[T], gains: Mapping[T, int], k: int) -> float:
    """Normalised discounted cumulative gain of the first `k` results.

    Each result contributes its gain divided by log2(rank + 1), so highly
    relevant results at the top count most. The sum is divided by the best
    achievable one, which makes 1.0 a perfect ranking. Gains are used linearly,
    as `trec_eval` does.
    """
    ideal = _dcg(sorted((gain for gain in gains.values() if gain > 0), reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return _dcg([max(gains.get(item, 0), 0) for item in ranking[:k]]) / ideal


def _dcg(gains: Sequence[int]) -> float:
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
